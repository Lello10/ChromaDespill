import numpy as np
import cv2
import sys
import argparse

def linearize_mask(image):
    h,w = image.shape
    return image.reshape((h*w))

# Data un immagine calcola la media e la deviazione standard dei canali
def image_stats(image, applyMask=False, lMeanM=1, lStdM=1, aMeanM=1, aStdM=1, bMeanM=1, bStdM=1):
    if applyMask == True:
        # Viene estratta la maschera dall'alpha dell'immagine 
        mask = linearize_mask(image[:,:,3])
        # Viene convertita l'immagine da RGB a L*a*b
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")

        # Dall'immagine vengo estratti i singoli canali
        (l, a, b) = cv2.split(image)
        
        # I canali vengono convertiti in un array lineare
        l = linearize_mask(l)
        a = linearize_mask(a)
        b = linearize_mask(b)

        # Viene presa solamente la porzione di immagine in cui è presente il soggetto
        l=l[mask!=0]
        a=a[mask!=0]
        b=b[mask!=0]
    else:
        # Viene convertita l'immagine da RGB a L*a*b
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        
        # Dall'immagine vengo estratti i singoli canali
        (l, a, b) = cv2.split(image)

    # Vengono calcolate: la devizione standard e la media

    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # le statiche vengono inserite nell'array, che sarà poi salvato su un file di testo
    stats = []
    stats.append(lMean * lMeanM)
    stats.append(lStd * lStdM)
    stats.append(aMean * aMeanM)
    stats.append(aStd * aStdM)
    stats.append(bMean * bMeanM)
    stats.append(bStd * bStdM)
    # return the color statistics

    stats = np.array(stats)

    return stats

def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
    description="Calibration for Transfer",
    prog='python3 transfe_calibrator.py')

    parser.add_argument("-t", "--target", help="path to target image file.",
                        type=str, default="TransferCalibrator/Input/Target.png")

    parser.add_argument("-s", "--source", help="path to source image file.",
                        type=str, default="TransferCalibrator/Input/Source.png")

    parser.add_argument("-o1", "--outpath1", help="path to source stats file.",
                        type=str, default="TransferCalibrator/Output/SourceStats.txt")

    parser.add_argument("-o2", "--outpath2", help="path to target stats file.",
                        type=str, default="TransferCalibrator/Output/TargetStats.txt")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    source = cv2.imread(args.source)
    target = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)

    sourceStats = image_stats(source)
    targetStats = image_stats(target, applyMask = True)

    np.savetxt(args.outpath1, sourceStats)
    np.savetxt(args.outpath2, targetStats)
