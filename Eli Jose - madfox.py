import numpy as np
import cv2


sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

img1=cv2.imread("madfox.jpg",0)
trainKP,trainDesc = sift.detectAndCompute(img1,None)





cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)




while(True):

	confirmacao = 0  #variavel para guardar se o circulo e a raposa foram encontrados


	# Capture frame-by-frame
	ret, frame = cap.read()
	# Convert the frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# A gaussian blur to get rid of the noise in the image
	gray = cv2.GaussianBlur(gray,(5,5),0)
	bordas_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)



###### deteccao da raposa  #######

	queryKP,queryDesc = sift.detectAndCompute(gray,None)
	matches = flann.knnMatch(queryDesc,trainDesc,k=2)

	goodMatch = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			goodMatch.append(m)

	MIN_MATCH_COUNT = 30
	if len(goodMatch) > MIN_MATCH_COUNT:
		confirmacao += 2
		tp=[]
		qp=[]

		for m in goodMatch:
			tp.append(trainKP[m.trainIdx].pt)
			qp.append(queryKP[m.queryIdx].pt)

		tp,qp=np.float32((tp,qp))

		H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)

		h,w = img1.shape
		trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
		queryBorder=cv2.perspectiveTransform(trainBorder,H)
		cv2.polylines(bordas_color,[np.int32(queryBorder)],True,(0,255,0),5)
	else:
		print "Raposa nao encontrada- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)


########  deteccao de circulo  #####

	circles = []
	circles = None
	circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=125,minRadius=5,maxRadius=60)

	if circles is not None:
		confirmacao += 3
		circles = np.uint16(np.around(circles))

		for i in circles[0,:]:
			
			# draw the outer circle
			# cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
			cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
			# draw the center of the circle
			cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)


	if confirmacao == 5:
		cv2.putText(bordas_color,'Imagem conjunta encontrada', (20,430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,170,0),2)

		if (circles[0][0][0]) > (queryBorder[0][0][0]):
			cv2.putText(bordas_color,'Circulo esta a direita da raposa', (20,460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,170,0),2)
			# ESQUERDA CONSIDERANDO UM OBSERVADOR OBSERVANDO O PAPEL REAL ONDE AS FIGURAS SE ENCONTRAM
		else:
			cv2.putText(bordas_color,'Circulo esta a esquerda da raposa', (20,460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,170,0),2)
			# ESQUERDA CONSIDERANDO UM OBSERVADOR OBSERVANDO O PAPEL REAL ONDE AS FIGURAS SE ENCONTRAM

	if confirmacao == 3:
		cv2.putText(bordas_color,'Circulo encontrado', (20,430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,170,0),2)
	if confirmacao == 2:
		cv2.putText(bordas_color,'Raposa encontrada', (20,430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,170,0),2)




	cv2.imshow('Detector de circulos',bordas_color)
	print("No circles were found")
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()