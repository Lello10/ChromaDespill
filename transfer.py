import cv2
import sys
import argparse
import numpy as np

from transfer_calibrator import image_stats

# Data un'immagine fa una ricalibrazione del colore utilizzando delle statistiche precedentemente calcolate
def color_transfer(target, sourceStats, targetStats, preserve_paper=False):
	#Vengono lette le statistiche e valorizzate le variabili
	lMeanSrc = sourceStats[0]
	lStdSrc = sourceStats[1]
	aMeanSrc = sourceStats[2]
	aStdSrc = sourceStats[3]
	bMeanSrc = sourceStats[4] 
	bStdSrc = sourceStats[5]

	lMeanTar = targetStats[0]
	lStdTar = targetStats[1]
	aMeanTar = targetStats[2]
	aStdTar = targetStats[3]
	bMeanTar = targetStats[4]
	bStdTar = targetStats[5]

	# Viene convertita l'immagine da RGB a L*a*b
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	# Dall'immagine vengo estratti i singoli canali
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)

	# Ad ogni canate e` sottratta la propria media
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	#Vengono ridimensionati i canali
	if preserve_paper:
		l = (lStdTar / lStdSrc) * l
		a = (aStdTar / aStdSrc) * a
		b = (bStdTar / bStdSrc) * b
	else:
		l = (lStdSrc / lStdTar) * l
		a = (aStdSrc / aStdTar) * a
		b = (bStdSrc / bStdTar) * b

	#  Ad ogni canate e` aggiunta la corrispettiva media della sorgente
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# I valori vengo limitati al range [0-255]
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# I canali vengono nuovamente uniti e l'immagine viene poi convertita in RGB
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	return transfer

def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
    description="Transfer",
    prog='python3 transfer.py')

    parser.add_argument("-ss", "--sourceStats", help="path to source stats file.",
    					type=str, default="TransferCalibrator/Output/SourceStats.txt")

    parser.add_argument("-st", "--targetStats", help="path to target stats file.",
   						 type=str, default="TransferCalibrator/Output/TargetStats.txt")

    parser.add_argument("-o", "--outpath", help="path to output transfer image.",
    					type=str, default="Transfer/Output/ImageTransfered.png")


    return parser.parse_args(args)


if __name__ == "__main__":
	args = parse_args()

	sourceStats = np.loadtxt(args.sourceStats).astype("float32")
	targetStats = np.loadtxt(args.targetStats).astype("float32")

	target = cv2.imread("TransferCalibrator/Input/Target.png", cv2.IMREAD_UNCHANGED)
	mask = target[:,:,3]

	result = color_transfer(target, sourceStats, targetStats)

	result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
	result[:,:,3] = mask
	
	cv2.imwrite(args.outpath,result)
	 


