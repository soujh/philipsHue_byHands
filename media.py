import mediapipe as mp
from cv2 import cv2 as cv2
from phue import Bridge 

b = Bridge('#Hub IP here') 

#Lumnosité max=254

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands #Contient toutes les méthodes pour l'analyse d'une main
hands=mpHands.Hands() 
mpDraw= mp.solutions.drawing_utils #cela va nous aider à encader (dessiner) notre main 

doigtsCoor=[(8,6),(12,10),(16,14),(20,18)] #Liste des points à comparer pour définir un doigt commme ouvert/fermer
pouceCoor=(4,2)


while True:
    success,img=cap.read()
    RGB_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #On doit convertir notre image en RGB pour qu'elle soit 'processable' par mediapipe
    results=hands.process(RGB_img)   #Process l'image
    landmarked_hand=results.multi_hand_landmarks #Ici on reconnait la main
    """
    print(landmarked_hand)
    > landmark {
        x: 0.3402962386608124
        y: 0.3880709111690521
        z: -0.029443103820085526
    } 
    """
    if landmarked_hand:
        handPoints=[]
        for landmark in landmarked_hand: #On itère sur chaque mains reconnues 
            mpDraw.draw_landmarks(img,landmark,mpHands.HAND_CONNECTIONS) #On déssine les
            for idx, lm in enumerate(landmark.landmark): #coordonnées pour chaque points
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h) #a la place des coordonnées on cherche à avoir le pixel depuis la droite
                """
                print(idx,cx,cy)
                >0 328 703
                14 397 675
                34 52 621
                20 503 572
                """
                handPoints.append((cx,cy)) #list des pixels pour chaque landmarks

        for point in handPoints:
            cv2.circle(img,point,7,(56,119,246),cv2.FILLED) #dessiner des cercle sur les points

        count=0
        for coor in doigtsCoor:
            if handPoints[coor[0]][1] < handPoints[coor[1]][1]: #Exemple pour l'index 
                """
                Exemple pour l'index :

                si handPoint[coor[8]][1] < handPoint[coor[6]][1] alors l'index est ouvert
            
                """ 
                count+=1
       
        if handPoints[pouceCoor[0]][0] > handPoints[pouceCoor[1]][0]: #Cas particulier pour le pouce
            count +=1

        cv2.putText(img,str(count),(150,150), cv2.FONT_HERSHEY_PLAIN,12,(0,108,255),12)
        for l in b.lights:
            l.brightness=50*count

    cv2.imshow("Hue by hands", img)
    cv2.waitKey(1)



"""
if __name__ == '__main__':
    main()

"""

