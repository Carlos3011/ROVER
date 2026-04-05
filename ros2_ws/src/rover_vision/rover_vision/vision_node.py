#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import pytesseract

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(String, 'detections', 10)
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_count = 0

        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir la cámara")

        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        # =======================================================
        # OCR (Letrero FIN) - Se ejecuta cada 15 frames
        # =======================================================
        if self.frame_count % 15 == 0:
            roi_top = frame[0:240, 0:640] 
            gray = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            try:
                texto_detectado = pytesseract.image_to_string(thresh, config='--psm 11').strip().upper()
                
                if "FIN" in texto_detectado:
                    msg_fin = String()
                    msg_fin.data = "fin,320,negro,N/A" 
                    self.publisher_.publish(msg_fin)
                    cv2.putText(frame, "LETRERO FIN ENCONTRADO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            except Exception as e:
                pass 

        # =======================================================
        # DETECCIÓN DE ROCAS POR COLOR
        # =======================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
        mask_red2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
        mask_red = mask_red1 + mask_red2

        mask_blue = cv2.inRange(hsv, np.array([100,150,50]), np.array([140,255,255]))
        mask_green = cv2.inRange(hsv, np.array([40,70,70]), np.array([80,255,255]))

        self.detect_color(mask_red, frame, "rojo")
        self.detect_color(mask_blue, frame, "azul")
        self.detect_color(mask_green, frame, "verde")

        # =======================================================
        # DETECCIÓN DEL PANEL DE MANTENIMIENTO
        # =======================================================
        # Filtro HSV para aislar las zonas grises
        mask_gray = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_gray:
            largest_gray = max(contours_gray, key=cv2.contourArea)
            if cv2.contourArea(largest_gray) > 5000: # Validación de tamaño mínimo
                x, y, w, h = cv2.boundingRect(largest_gray)
                cx = int(x + w/2)
                
                # Mapeo de pixeles en Y a altura en mm (ajustar constantes tras pruebas físicas)
                altura_estimada_mm = int(1000 - (y * 2)) 
                
                msg_panel = String()
                msg_panel.data = f"panel,{cx},{altura_estimada_mm}"
                self.publisher_.publish(msg_panel)
                
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
                cv2.putText(frame, f"PANEL: {altura_estimada_mm}mm", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        try:
            cv2.imshow("Vision - Fat Rat", frame)
            cv2.waitKey(1)
        except:
            pass

    def detect_color(self, mask, frame, color_name):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filtro para evadir ruido
        if area > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = int(x + w/2)
            
            # Estimación de volúmenes
            if area < 1500: tamano = "5cm3"
            elif area < 3000: tamano = "7cm3"
            elif area < 5000: tamano = "10cm3"
            else: tamano = "12cm3"

            # NUEVA LÓGICA DE TEXTURA (Varianza Laplaciana)
            # Extraemos la Región de Interés (ROI) de la roca
            roi_color = frame[y:y+h, x:x+w]
            
            # Prevenir errores si el ROI se sale del cuadro
            if roi_color.size > 0:
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                
                # Calculamos la varianza del Laplaciano
                varianza = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                
                # Umbral empírico: Deberás calibrar este número "300"
                if varianza > 300:
                    textura = "rugosa"
                else:
                    textura = "lisa"
            else:
                textura = "no_determinada"

            msg = String()
            msg.data = f"roca,{cx},{color_name},{tamano},{textura}"
            self.publisher_.publish(msg)

            # Dibujamos en pantalla para monitoreo
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{color_name} {tamano} {textura}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
