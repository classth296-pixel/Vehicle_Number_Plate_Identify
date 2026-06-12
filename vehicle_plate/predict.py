    
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\detect\train-10\weights\best.pt')
    results = model(r'License_plate\test_images\pexels-garvin-st-villier-719266-5270094.jpg')
    results[0].show()  # popup window
    # OR save to file instead:
    results[0].save(filename='output.jpg')