"""
option pour utiliser mask_generaot.py
    # Afficher l'image chargée
    show_image(image)
    # Générer les masques à partir de l'image
    masks = generate_masks(mask_generator, image)
    # Afficher les masques générés
    display_generate_masks(image, masks)
    # Afficher le masque correspondant à l'indice 12
    display_index_mask(masks, 12)
    # Fusionner les masques correspondant aux indices 12, 21 et 10, puis les afficher
    merged_mask = merge_and_display_binary_masks(masks, [12, 21, 10])
    # Appliquer la couleur des masques fusionnés à l'image d'origine et afficher le résultat
    apply_mask_color_and_display(image_path, merged_mask)
    # Détecter les contours dans le masque fusionné et les afficher
    detect_and_display_contours(merged_mask)
    # Dessiner les contours détectés sur l'image d'origine et enregistrer le résultat
    draw_contours_original_image_save(image, merged_mask)
"""

import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch


def generate_masks(mask_generator, image):
    """
    Génère tous les masques à partir de l'image à l'aide du générateur de masques SAM.
    
    Args:
    - mask_generator : Générateur de masques SAM
    - image : Image à partir de laquelle générer les masques
    
    Returns:
    - masks : Liste d'objets de masque renvoyés par SAM
    """
    # Générer les masques à partir de l'image
    masks = mask_generator.generate(image)
    
    return masks

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def display_generate_masks(image, masks):
    """
    Affiche l'image avec les masques générés.
    
    Args:
    - image : Image à afficher
    - masks : Liste d'objets de masque à afficher
    """
    # Afficher l'image avec les masques générés
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def display_index_mask(masks, index):
    """
    Affiche un masque à partir de l'objet de masque renvoyé par SAM.

    Args:
    - masks : Liste d'objets de masque renvoyés par SAM
    - index : Indice du masque
    """
    # Convertir le masque en image PIL
    image_mask_pil = Image.fromarray(masks[index]['segmentation'])
    
    # Afficher l'image du masque à l'aide de Matplotlib
    plt.imshow(image_mask_pil, cmap='gray')
    plt.axis('off')
    plt.show()

def merge_and_display_binary_masks(masks, indices):
    """
    Fusionne plusieurs masques ensemble et affiche le résultat.

    Args:
    - masks : Liste d'objets de masque renvoyés par SAM
    - indices : Liste des indices des masques à fusionner
    
    Returns:
    - merged_mask : Masque fusionné (numpy array)
    """
    # Vérifier que les indices sont valides
    if any(idx < 0 or idx >= len(masks) for idx in indices):
        print("Indices de masque invalides.")
        return None
    
    # Fusionner les masques
    merged_mask = masks[indices[0]]['segmentation']
    for idx in indices[1:]:
        merged_mask = merged_mask | masks[idx]['segmentation']

    # Afficher l'image du masque fusionné
    plt.imshow(merged_mask, cmap='gray')
    plt.axis('off')
    plt.show()

    return merged_mask

def apply_mask_color_and_display(image_path, merged_mask):
    """
    Applique un masque à une image, convertit l'image résultante en format PNG et l'affiche dans le notebook.

    Args:
    - image_path : Chemin vers l'image
    - merged_mask : Masque à appliquer (numpy array)
    """
    mask_tensor = merged_mask
    # Charger l'image
    image_originale = cv2.imread(image_path)

    # Binariser le masque
    mask_array = (mask_tensor > 0.5).astype(np.uint8) * 255

    # Appliquer le masque à l'image
    masked_image = cv2.bitwise_and(image_originale, image_originale, mask=mask_array)

    # Afficher l'image à l'aide de Matplotlib
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))  # Convertir BGR en RGB pour Matplotlib
    plt.axis('off')
    plt.show()

def detect_and_display_contours(merged_mask):
    """
    Detect contours in a mask and display them using Matplotlib.

    Args:
    - merged_mask : Mask from which to detect contours (numpy array)
    """
    mask_tensor = merged_mask
    # Convert the boolean mask to an 8-bit binary mask
    mask_array = (mask_tensor > 0.5).astype(np.uint8) * 255

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    # Create an empty image to draw contours

    # Draw the contours on the image
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), 2)

    # Display the image with detected contours using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(contour_mask, cmap='gray') # Use the contour_mask directly
    plt.axis('off')
    plt.show()


def draw_contours_original_image_save(image, merged_mask):
    """
    Dessine les contours sur une copie de l'image originale et affiche l'image avec les contours à l'aide de Matplotlib.
    Enregistre également l'image avec les contours dans un dossier spécifié.
    

    Args:
    - image : Image originale (numpy array)
    - mask_tensor : Masque à partir duquel détecter les contours (numpy array)
    - folder_path : Chemin du dossier pour enregistrer l'image
    """
    folder_path = 'assets/'
    mask_tensor = merged_mask
    # Convertir le masque booléen en masque binaire 8 bits
    mask_array = (mask_tensor > 0.5).astype(np.uint8) * 255

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur une copie de l'image originale
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 2)

    # Afficher l'image avec les contours à l'aide de Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_contours)
    plt.axis('off')
    plt.show()

    # Enregistrer l'image avec les contours dans le dossier spécifié
    image_path = os.path.join(folder_path, 'image_enregistree.png')
    cv2.imwrite(image_path, cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)) 