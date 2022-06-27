import cv2
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter
import sys
import argparse

import itertools
import threading
import time

from transfer import color_transfer

CHROMA_RADIUS = 15
BORDER_SIZE = 2
SENSITIVITY = 0.95

#Viene creata una rampa che va da 1 (il colore più "vicino" a clr) a 0 (il colore più "lontano")
def linear_ramp(mask, start, delta):
    # I valori vengo limitati al range [0-1]
    return np.clip((mask - start) / delta, 0.0, 1.0)

# Data un'immagine e una palette di colori calcola la maschera di composizione
def chroma_mask_palette_ramped(imageYCbCr,palettePath):
    # Vengono salvate le dimensioni dell'immagine
    h,w,c = imageYCbCr.shape
    # Viene caricata la palette
    palette = np.loadtxt(palettePath)
    # Viene creata una maschera, cui i valori hanno valore 0, delle stesse dimensioni dell'immagine 
    mask = np.zeros([h,w], float)
    #Per ogni colore della palette
    for ii, clr in enumerate(palette):
        # Viene calcolata la normale per ogni valore, dell'array a cui viene sottratto il colore
        m2 = np.linalg.norm(imageYCbCr - clr, axis=2, ord=2)
        # Viene restituito il complementare
        m3 = 1.0 - linear_ramp(m2, CHROMA_RADIUS * 1.25, CHROMA_RADIUS * 0.50)
        # Per ogni valore dei due array viene preso il piu' grande
        mask = np.maximum(mask, m3)
    return 1-mask

# Data un immagine corregge gli aloni di verde
def despill2(imageBGR, sensitivity, doRamp, rampDelta=24):
    # Viene creata una copia dell'immagine
    np_image = np.array(imageBGR)
    # Dall'immagine vengo estratti i due canali
    r = np_image[:,:,2]
    g = np_image[:,:,1]
    b = np_image[:,:,0]

    # Viene calcolata la media tra il canale rosso e il canale blu
    replace_val = b/2 + r/2
    # Viene definita la soglia pesata
    weighted_threshold = np.clip(replace_val * sensitivity, 0, 255)
    # Viene valorizzato l'array delle condizioni
    cond = linear_ramp(g.astype(float), weighted_threshold, rampDelta) if doRamp else (g > weighted_threshold).astype(float)
    # Viene valorizzato l'array di valori con i valori del verde corretti
    replace_g = np.minimum(replace_val, g)
    # Se la condizione e` positiva il valore del verde, in un pixel, viene sostituito con quello corretto
    np_image[:,:,1] = cond * replace_g + (1-cond) * g
    return cond, np_image

# Generate a checkered background
def checkboard(image):
    h, w, _ = image.shape
    T = 16
    for x in range(0, w, T):
        for y in range(0, h, T):
            color = (255,255,255) if ((x+y)//T) % 2 == 0 else (0,0,128)
            cv2.rectangle(image, (x,y), (x+T,y+T), color, -1)
    return image

def resizeImg(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

# Compone due immagini utilizzando la maschera di composizione
def cv2_compose(foreground, background, alphamask, posx, posy , scale):
    # Viene ricordato il vecchio datatype
    old_type = background.dtype
    # alphamask viene composta da 3 canali tutti uguali
    alphamask = np.stack([alphamask]*3, axis=2)
    # Viene moltiplicata la maschera all'immagine convertita da uint8 a float
    foreground = cv2.multiply(alphamask, foreground.astype(float))
    # La maschera e l'immagine vengono ridimensionate allo stesso modo
    foreground = resizeImg(foreground,scale)
    alphamask = resizeImg(alphamask,scale)
     # Vengono salvate le dimensioni dell'immagine ridimensionata
    fh,fw,_ = foreground.shape
    # Viene convertito da uint8 a float
    background = background.astype(float)
    # La maschera viene moltiplicata all'immagine di background nella posizione desiderata
    background[posy:(posy+fh),posx:(posx+fw)] = cv2.multiply(1.0 - alphamask, background[posy:(posy+fh),posx:(posx+fw)])
    # Viene aggiunto il foreground sul background
    background[posy:(posy+fh),posx:(posx+fw)] = cv2.add(background[posy:(posy+fh),posx:(posx+fw)],foreground)
    # L'immagine viere riportata al vecchio datatype
    outImage = background.astype(old_type)
    return outImage

def applyChromaDespill(imageBGR, backgroundBGR, palettePath, posx, posy, scale, sourceStats, targetStats, borderSz, sensitivity, applyDespill, addBorder, doRamp, applyTransfer):
    # Apply chroma key cancellation and remove the green screen
    imageYCbCr = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YCR_CB)
    chromaMask = chroma_mask_palette_ramped(imageYCbCr,palettePath)

    if addBorder:
        erosion_kernel = np.array([ [0,1,0], [1,1,1], [0,1,0] ], np.uint8)
        chromaMask = cv2.erode(chromaMask, erosion_kernel)
        borderMask = np.clip(gaussian_filter(chromaMask, sigma=borderSz), 0.0, 1.0)
        alphaMask = np.minimum(chromaMask, borderMask)
    else:
        alphaMask = chromaMask
        borderMask = np.array([[1.0]])
    alphaMask = chromaMask

    # Apply despill
    despMask, despilled = despill2(imageBGR, sensitivity, doRamp) # borderMask if addBorder else
    despMask = np.minimum(despMask, chromaMask)

    # compose with background
    srcCompose = despilled if applyDespill else imageBGR
    if applyTransfer:
        srcCompose = color_transfer(srcCompose, sourceStats, targetStats)
    result = cv2_compose(srcCompose, backgroundBGR, alphaMask, posx, posy, scale )

    return alphaMask, borderMask, despMask, result


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

def videoComposer(args):
    global done
    posx = args.posx
    posy = args.posy
    scale = args.scale
    palettePath = args.palettePath
    sourceStats = np.loadtxt(args.sourceStats).astype("float32")
    targetStats = np.loadtxt(args.targetStats).astype("float32")
    cap = cv2.VideoCapture(args.video)
    newBackground = cv2.imread(args.background)
    #newBackground = checkboard(np.zeros( (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8))
    #newBackground = np.zeros( (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
    h , w , _ = newBackground.shape
    frame_width = w
    frame_height = h
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_filename = args.outpath
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    print(f"{output_filename}  {frame_height}x{frame_width}@{fps} frames:{frame_count}")
    
    while cap.isOpened():
        done = False
        t = threading.Thread(target=animate)
        t.start()
        ret, frame = cap.read()
        if not ret:
            break

        _,_,_,procFrame = mask, borderMask, despMask, image = applyChromaDespill(frame, newBackground, palettePath,  posx, posy, scale, 
                                                                                sourceStats, targetStats, BORDER_SIZE, SENSITIVITY,
                                                                                applyDespill=True, addBorder=True, doRamp=True, applyTransfer = True)
        out.write(procFrame)

    done = True
    cap.release()
    out.release()
    #print('\nOk.')
    del cap, out

def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Calibration for ChromaDespill",
        prog='python3 Calibrator.py')

    parser.add_argument("-v", "--video", help="path to video file.",
                        type=str, default="ChromaDespill/Input/Video/prova.avi")

    parser.add_argument("-b", "--background", help="path to background file.",
                        type=str, default="ChromaDespill/Input/Background/provaBackground.png")

    parser.add_argument("-ss", "--sourceStats", help="path to source stats file.",
                        type=str, default="TransferCalibrator/Output/SourceStats.txt")

    parser.add_argument("-st", "--targetStats", help="path to target stats file.",
                        type=str, default="TransferCalibrator/Output/TargetStats.txt")

    parser.add_argument("-o", "--outpath", help="file path to write output."
                        " format: <path>.<format(avi,mp4)>",
                        type=str, default="ChromaDespill/Output/provaFinal.avi")

    parser.add_argument("-p", "--palettePath", help="path to palette file."
                        " format: <path>.<format(txt)>",
                        type=str, default="Calibrator/Output/palette.txt")

    parser.add_argument("-x", "--posx", help="offset axis x.",
                        type=int, default="350")

    parser.add_argument("-y", "--posy", help="offset axis y.",
                        type=int, default="0")

    parser.add_argument("-s", "--scale", help="percent of scale.",
                        type=int, default="241")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    videoComposer(args)
