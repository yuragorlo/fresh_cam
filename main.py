# import pytesseract
# img = '/home/mnemonic/work/digits/real_source_2/21.jpg'
# custom_config = r'--oem 3 --psm 6 outputbase digits'
# print(pytesseract.image_to_string(img, config=custom_config))

# read hid device python
# https://www.ontrak.net/pythonhidapi.htm
# https://pypi.org/project/hid/

# full screen opencv
# https://answers.opencv.org/question/198479/display-a-streamed-video-in-full-screen-opencv3/

import pyzbar.pyzbar as pyzbar
import numpy as np
import mysql.connector
import threading
import cv2
import time
import logging
from datetime import datetime
from contextlib import closing


def create_table(config):
    db_conn = mysql.connector.connect(**config)
    with closing(db_conn) as conn:
        with conn.cursor() as cursor:
            cursor.execute('''CREATE TABLE IF NOT EXISTS qr_table 
            (id INT NOT NULL AUTO_INCREMENT, 
            data VARCHAR(100) NOT NULL, 
            time TIMESTAMP NOT NULL, 
            route VARCHAR(10) NOT NULL, 
            check_flag INT NOT NULL, 
            PRIMARY KEY (id));''')
            conn.commit()


def push_to_db(qr_dict, config):
    db_conn = mysql.connector.connect(**config)
    with closing(db_conn) as conn:
        with conn.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(qr_dict))
            columns = ', '.join(qr_dict.keys())
            sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % ('qr_table', columns, placeholders)
            cursor.execute(sql, list(qr_dict.values()))
            conn.commit()
            rootLogger.info(f'{qr_dict} data have been written to db')


def process_new_qr(decodedObjects, STOCK_LAST_DATA, camera_label):
    min_size_stock_last_data = 3
    max_size_stock_last_data = 7
    default_check_flag = 0
    current_data_list = [x.data.decode("utf-8") for x in decodedObjects]

    # Push new data to db
    if not any(item in current_data_list for item in STOCK_LAST_DATA):
        for obj in decodedObjects:
            # Init dict with current data
            now = datetime.now()
            new_data = {'data': obj.data.decode("utf-8"),
                        'time': now.strftime("%Y/%m/%d %H:%M:%S"),
                        'route': camera_label,
                        'check_flag': default_check_flag}
            # Insert new qr code with time registration to qr_table qr database
            push_to_db(new_data, DB_CONF)

    # Add new data to STOCK_LAST_DATA
    for obj in decodedObjects:
        if obj.data.decode("utf-8") not in STOCK_LAST_DATA:
            STOCK_LAST_DATA.append(obj.data.decode("utf-8"))
            rootLogger.info(f'{obj.data.decode("utf-8")} have been append to STOCK_LAST_DATA')

    # Clean STOCK_LAST_DATA and pass only min_size_stock_last_data elements
    if len(STOCK_LAST_DATA) > max_size_stock_last_data:
        STOCK_LAST_DATA = STOCK_LAST_DATA[-min_size_stock_last_data:]
        rootLogger.info(f'Delete part of data from stock. Current stock: {STOCK_LAST_DATA}')

    return STOCK_LAST_DATA


def add_qr_to_frame(frame, decodedObjects):
    for decodedObject in decodedObjects:
        # Draw square around qr
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        # Number of points in the convex hull
        n = len(hull)
        # Draw the hull
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 0, 255), 3)

        # Show text
        x = decodedObject.rect.left
        y = decodedObject.rect.top
        barCode = str(decodedObject.data.decode("utf-8"))
        cv2.putText(frame, barCode, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def process_stream(STREAM_URL, camera_label, DB_CONF):
    try:
        # Create mysql table for qr data if not exist
        create_table(DB_CONF)

        # Get the webcam or stream:
        if STREAM_URL == '':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(STREAM_URL)

        # Resize cap example size 1024.0 x 768.0; 1280.0 x 1024.0
        cap.set(3, 640)
        cap.set(4, 480)
        time.sleep(2)

        # Init stock - short memory for qr data
        STOCK_LAST_DATA = []

        # Endless cycle
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Fix time
            last_time = time.time()
            # Our operations on the frame come here
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Processing QR in frame
            decodedObjects = pyzbar.decode(im)
            if decodedObjects:
                # Update stock and push to db if not in stock
                STOCK_LAST_DATA = process_new_qr(decodedObjects, STOCK_LAST_DATA, camera_label)
                # Add square around qr and text with data on frame
                add_qr_to_frame(frame, decodedObjects)
            # Display FPS:
            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Display the resulting frame
            cv2.imshow(camera_label + '_frame', frame)
            # Control keys
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):  # wait for 's' key to save
                cv2.imwrite('Capture.png', frame)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        rootLogger.error(e)


# Global variables
# Example "rtsp://192.168.1.2:8080/out.h264" if empty() - use front notebook camera
RIGHT_STREAM = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov'
LEFT_STREAM = ''
# Side labels. There are 'names' of cameras. We have 2 cameras in system: 'left' and 'right' now
RIGHT_LABEL = 'right'
LEFT_LABEL = 'left'
# DB config. You should create qr database, but qr_table table will create automatically if it not exist
DB_CONF = {
    'host': "localhost",
    'user': "mysql_user",
    'password': "mysql_password",
    'database': "qr"}

if __name__ == '__main__':

    # Logger
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [function %(funcName)s] [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger('root')
    rootLogger.setLevel(logging.INFO)
    fileHandler_root = logging.FileHandler("log.log")
    fileHandler_root.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler_root)

    # Process streams
    thread_right = threading.Thread(target=process_stream, args=(RIGHT_STREAM, RIGHT_LABEL, DB_CONF,))
    thread_left = threading.Thread(target=process_stream, args=(LEFT_STREAM, LEFT_LABEL, DB_CONF,))
    thread_right.start()
    rootLogger.info('thread_right have been started')
    thread_left.start()
    rootLogger.info('thread_left have been started')
    thread_right.join()
    rootLogger.info('thread_right have been stopped')
    thread_left.join()
    rootLogger.info('thread_left have been stopped')
