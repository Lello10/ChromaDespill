import cv2
import numpy as np
import random
import sys
import argparse
import glob
import itertools
import threading
import time


# linearize into an array all pixels from a bitmap
def linearize_pixels(image):
    h,w,c = image.shape
    return image.reshape((h*w, c))


# Select randomly a palette of colors, s.t. these colors with CHROMA_RADIUS
# radius can be used an keys in chroma removal. Maximize removal from the
# backgroundYCbCr, and minimize removal from the foregroundYCbCr image
CHROMA_RADIUS = 15
BACKGROUND_MIN_Y_VALUE = 25
def select_chroma_colors(backgroundYCbCr, foregroundYCbCr):
    random.seed(12345)
    # Si ridimensionano le immagini in un array lineare
    linB = linearize_pixels(backgroundYCbCr)
    linF = linearize_pixels(foregroundYCbCr)
    # Dal background vengono rimossi tutti i pixel neri
    linB = linB[ linB[:,0] > BACKGROUND_MIN_Y_VALUE ].astype(float)
    init_colors = len(linB)

    colors = []
    iters = 0
    while len(linB) > 0 and len(colors) < 30 and iters < 1000:
        iters += 1
        # Si prendono NUM_TEST_SAMPLES colori, viene selezionato quello con il cluster più grande
        NUM_TEST_SAMPLES = 16
        candidates = []
        for ii in range(NUM_TEST_SAMPLES):
            # Si prende un verde dal background in modo casuale
            clr = linB[random.randint(0, len(linB)-1)]
            # Viene calcolata la normale per ogni valore, dell'array a cui viene sottratto il verde preso
            distB = np.linalg.norm(linB - clr, axis=1, ord=2)
            distF = np.linalg.norm(linF - clr, axis=1, ord=2)
            # Si verifica se ogni valore sia minore della costante
            inClusterB = (distB < CHROMA_RADIUS)
            inClusterF = (distF < CHROMA_RADIUS)
            # Si fa la sommatoria dei valori dell'array (true = 1, false = 0)
            countB, countF = np.sum(inClusterB), np.sum(inClusterF)
            # Viene calcolato un punteggio per il colore scelto
            ratioBF = countB / (countF + 1)
            # Se il punteggio è superiore a 10, il colore viene aggiunto alla lista dei candidati per la palette
            if ratioBF > 10:
                candidates.append( (ratioBF, clr) )

        if len(candidates) == 0:
            break
        # I candidati sono ordinati per punteggio
        candidates = sorted(candidates, key=lambda x:x[0])
        # Viene selezionato il colore con il punteggio maggiore
        clr = candidates[-1][1]
        # Vengono rimossi i colori vicini a clr 
        distB = np.linalg.norm(linB - clr, axis=1, ord=2)
        inClusterB = (distB < CHROMA_RADIUS)
        linB = linB[ ~inClusterB ]
        # Il colore viene aggiunto alla palette
        colors.append(list(clr))
    return np.array(sorted(colors, reverse=True))

def loadAndComposeData(fgPath,bgPath):
    composedF = np.array([])
    composedB = np.array([])
    for filenameF, filenameB in zip(glob.glob(fgPath), glob.glob(bgPath)):
        if len(composedF.tolist()) == 0  and len(composedB.tolist()) == 0:
            composedF = cv2.imread(filenameF)
            composedB = cv2.imread(filenameB,cv2.IMREAD_UNCHANGED)
        else:
            background = cv2.imread(filenameB,cv2.IMREAD_UNCHANGED)
            foreground = cv2.imread(filenameF)
            composedB = cv2.vconcat([composedB,background])
            composedF = cv2.vconcat([composedF,foreground])

    mask = composedB
    mask = mask[:,:,3]<20
    
    composedB[ mask ] = (0,0,0,0)
    composedF[ ~mask ] = (0,0,0)
    composedB = cv2.cvtColor(composedB, cv2.COLOR_BGRA2BGR)
    composedB = cv2.cvtColor(composedB, cv2.COLOR_BGR2YCR_CB)
    composedF = cv2.cvtColor(composedF, cv2.COLOR_BGR2YCR_CB)
    return composedB,composedF

def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Calibration for ChromaDespill",
        prog='python3 Calibrator.py')

    parser.add_argument("-f", "--foreground", help="path to foreground file.",
                        type=str, default="Calibrator/Input/Image/*.png")

    parser.add_argument("-b", "--background", help="path to background file.",
                        type=str, default="Calibrator/Input/Background/*.png")

    parser.add_argument("-o", "--outpath", help="file path to write output to."
                        " format: <path>.<format(txt)>",
                        type=str, default="Calibrator/Output/palette.txt")

    return parser.parse_args(args)

if __name__ == "__main__":
    global done
    done = False
    args = parse_args()
    t = threading.Thread(target=animate)
    t.start()
    fgPath = args.foreground
    bgPath = args.background
    background, foreground = loadAndComposeData(fgPath,bgPath)
    paletteYCbCr = select_chroma_colors(background, foreground)
    np.savetxt(args.outpath, paletteYCbCr)
    
    done = True

