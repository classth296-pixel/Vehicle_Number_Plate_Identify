from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(r'runs\detect\train-10\weights\best.pt')
    
    cap = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated = results[0].plot()  
        
        cv2.imshow('License Plate Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()