"""
Script para crear un plano de planta sintético para probar el Django
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_floor_plan_image(width=800, height=600, output_path='plano_prueba.png'):
    """
    Crea una imagen sintética que parece un plano de planta arquitectónico
    """
    # Crear imagen blanca
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Colores
    wall_color = (0, 0, 0)  # Negro para paredes
    floor_color = (240, 240, 240)  # Gris claro para pisos

    # Grosor de paredes
    wall_thickness = 8

    # Dibujar perímetro exterior (casa)
    margin = 50
    outer_rect = [margin, margin, width-margin, height-margin]
    draw.rectangle(outer_rect, outline=wall_color, width=wall_thickness)

    # Dibujar habitaciones

    # Habitación 1: Dormitorio (arriba izquierda)
    bedroom = [margin, margin, width//2 - 20, height//2 - 20]
    draw.rectangle(bedroom, outline=wall_color, width=wall_thickness)
    draw.text((bedroom[0] + 50, bedroom[1] + 50), "BEDROOM", fill=(100, 100, 100))

    # Habitación 2: Baño (arriba derecha)
    bathroom = [width//2 + 20, margin, width-margin, height//3]
    draw.rectangle(bathroom, outline=wall_color, width=wall_thickness)
    draw.text((bathroom[0] + 30, bathroom[1] + 30), "BATHROOM", fill=(100, 100, 100))

    # Habitación 3: Cocina (centro derecha)
    kitchen = [width//2 + 20, height//3 + 20, width-margin, 2*height//3]
    draw.rectangle(kitchen, outline=wall_color, width=wall_thickness)
    draw.text((kitchen[0] + 40, kitchen[1] + 40), "KITCHEN", fill=(100, 100, 100))

    # Habitación 4: Sala (abajo izquierda)
    living = [margin, height//2 + 20, width//2 - 20, height-margin]
    draw.rectangle(living, outline=wall_color, width=wall_thickness)
    draw.text((living[0] + 40, living[1] + 40), "LIVING ROOM", fill=(100, 100, 100))

    # Habitación 5: Comedor (abajo derecha)
    dining = [width//2 + 20, 2*height//3 + 20, width-margin, height-margin]
    draw.rectangle(dining, outline=wall_color, width=wall_thickness)
    draw.text((dining[0] + 30, dining[1] + 30), "DINING", fill=(100, 100, 100))

    # Dibujar puertas (espacios en las paredes)
    door_width = 40
    door_color = (200, 200, 200)

    # Puerta entre dormitorio y sala
    draw.rectangle([width//2 - 30, height//2 - 10, width//2 - 10, height//2 + 10], fill='white')

    # Puerta entre cocina y comedor
    draw.rectangle([width//2 + 15, 2*height//3 + 10, width//2 + 25, 2*height//3 + 30], fill='white')

    # Agregar medidas
    draw.text((width//2 - 100, height - 30), "FLOOR PLAN - TEST", fill=(150, 150, 150),
              font=None)
    draw.text((margin + 10, height - 30), "Scale: 1:100", fill=(150, 150, 150))

    # Agregar algunos muebles simples
    # Cama en dormitorio
    bed_x, bed_y = bedroom[0] + 100, bedroom[1] + 120
    draw.rectangle([bed_x, bed_y, bed_x + 80, bed_y + 120], outline=(150, 150, 150), width=2)

    # Inodoro en baño
    toilet_x, toilet_y = bathroom[0] + 80, bathroom[1] + 80
    draw.ellipse([toilet_x, toilet_y, toilet_x + 40, toilet_y + 40], outline=(150, 150, 150), width=2)

    # Mesa en comedor
    table_x, table_y = dining[0] + 60, dining[1] + 60
    draw.ellipse([table_x, table_y, table_x + 100, table_y + 80], outline=(150, 150, 150), width=2)

    # Sofá en sala
    sofa_x, sofa_y = living[0] + 80, living[1] + 120
    draw.rectangle([sofa_x, sofa_y, sofa_x + 120, sofa_y + 40], outline=(150, 150, 150), width=2)

    # Guardar imagen
    img.save(output_path)
    print(f"Plano de prueba creado: {output_path}")
    print(f"Tamaño: {width}x{height}")
    return output_path

if __name__ == "__main__":
    # Crear plano de prueba
    plano = create_floor_plan_image(
        width=800,
        height=600,
        output_path='plano_prueba_django.png'
    )

    # También crear uno más grande
    plano_hd = create_floor_plan_image(
        width=1200,
        height=900,
        output_path='plano_prueba_HD.png'
    )

    print("\n✅ Planos de prueba creados:")
    print("  - plano_prueba_django.png (800x600)")
    print("  - plano_prueba_HD.png (1200x900)")
    print("\nPuedes usar estas imágenes para probar el Django en:")
    print("  http://localhost:8001/")
