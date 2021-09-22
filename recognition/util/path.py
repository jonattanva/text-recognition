# coding: utf-8
import os
import csv
import json
import numpy as np

from PIL import ImageFont


def create_folder(path):
    """Crea una nueva carpeta"""
    os.makedirs(path, exist_ok=True)


def create_folder_if_not_exists(filename):
    """Crea una carpeta solo si no existe"""
    if not exists(dirname(filename)):
        create_folder(dirname(filename))


def resolve(path):
    """Convierte la ruta en una ruta absoluta"""
    root = os.path.join(os.path.dirname(__file__), os.pardir)
    root = os.path.join(root, os.pardir)
    root = os.path.join(root, path)
    root = os.path.abspath(root)
    return root


def listdir(path):
    """Obtiene todos los archivos que contengan el directorio"""
    return os.listdir(resolve(path))


def dirname(path):
    """Obtiene el nombre del directorio de una ruta"""
    return os.path.dirname(path)


def extension(path):
    """Obtiene la extensión del archivo"""
    _, ext = os.path.splitext(path)
    return ext


def exists(path):
    """Verifica que la ruta exista"""
    return os.path.exists(path)


def basename(path):
    """Devuelve el nombre final de la ruta"""
    return os.path.basename(path)


def name(path):
    """Obtiene la ruta del archivo"""
    filename, _ = os.path.splitext(path)
    return filename


def get_class_name():
    """Carga el nombre de las clases desde un archivo json"""
    return load_json(filename=resolve('resources/language.json'))


def get_chars():
    """Carga los caracteres desde un archivo json"""
    return load_json(filename=resolve('resources/chars.json'))


def get_anchors():
    """Carga los anclajes de un archivo txt"""
    with open(resolve('resources/anchors.txt'), mode='r') as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape([9, 2])


def get_fonts(size=14):
    """Carga todas las fuentes disponibles"""
    return [ImageFont.truetype(resolve('resources/fonts') + '/' + value, size=size) for value in
            listdir('resources/fonts') if value.endswith('.ttf')]


def load_json(filename):
    """Carga la información de un archivo de tipo json"""
    with open(filename, mode='r') as f:
        return json.load(f)


def load_txt(filename):
    """Carga el texto de un archivo de tipo txt"""
    with open(filename, 'r') as f:
        content = f.read()
    return content


def load_csv(filename):
    """Carga la información de un archivo tipo csv"""
    rows = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    rows = np.array(rows)
    return rows
