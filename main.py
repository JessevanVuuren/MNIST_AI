from NN_network import *
from NN_layers import *

from keras.utils import to_categorical
from keras.datasets import mnist

import numpy as np
import random
import pygame
import time
import math


WIDTH = 900
HEIGHT = 700
FPS = 60


class MainScreen:
    def __init__(self, width, height, fps, title, font) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.fps = fps

        pygame.init()
        pygame.font.init()

        self.clock = pygame.time.Clock()
        self.fontSystem = pygame.font.SysFont(font, 30)
        self.screen = pygame.display.set_mode((width, height))

    
    def render_text(self, text, x, y):
        text = self.fontSystem.render(text, True, "white")
        self.screen.blit(text, (x, y))


    def update_screen(self):
        pygame.display.flip()

    def in_bounding_box(self, rect):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return (mouse_x > rect[0] and mouse_x < rect[0] + rect[2] and
                mouse_y > rect[1] and mouse_y < rect[1] + rect[3])

    def render_button(self, text, rect, callback, color="#525252", highlight="#777777"):
        if (self.in_bounding_box(rect)):
             color = highlight
             if (pygame.mouse.get_pressed()[0]):
                callback()

        pygame.draw.rect(self.screen, color, rect, 0, 2)
        text_length = len(text) * 10.5
        self.render_text(text, rect[0] + (rect[2] / 2) - text_length / 2, rect[1] + (rect[3] / 2)- 9)
    
    def get_delta_time(self):
        return self.clock.tick(self.fps) / 1000


class Canvas:
    def __init__(self, screen, row, column, size, padding) -> None:
        self.pixel_array = []
        self.screen = screen
        self.column = column
        self.row = row
        self.size = size
        self.padding = padding
        self.clear_screen()


    def clear_screen(self):
        self.pixel_array = [(82,82,82) for _ in range(self.column * self.row)]

    def render_canvas(self, offset_x, offset_y):
        for y in range(self.row):
            for x in range(self.column):
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_down = pygame.mouse.get_pressed()[0]

                pos_x = x * (self.size + self.padding) + offset_x
                pos_y = y * (self.size + self.padding) + offset_y

                distance = math.sqrt(math.pow(mouse_x - pos_x, 2) + math.pow(mouse_y - pos_y, 2))
                if (distance < 25 and mouse_down):    
                    self.pixel_array[y * self.row + x] = (255, 255, 255)

                pygame.draw.rect(self.screen.screen, self.pixel_array[y * self.row + x], pygame.Rect(pos_x, pos_y, self.size, self.size), 0, 2)

    def get_pixel_array(self):
        arr = []
        for i in range(self.column):
            arr_t = []
            for j in range(self.row):
                pixel = self.pixel_array[i * self.column + j]
                if (pixel == (82, 82, 82)): arr_t.append(0)
                else: arr_t.append(1)
        

            arr.append(arr_t)
    
        return np.array([arr])
    
    def set_pixel_array(self, arr):
        new_arr = []
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                if (arr[0][i, j] > 0):
                    new_arr.append((255, 255, 255))
                else:
                    new_arr.append((82, 82, 82))
        self.pixel_array = new_arr

    def is_screen_clear(self):
        return (255, 255, 255) not in self.pixel_array


def preprocess_data(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255

    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 50)


network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

load_model(network, "mnist_model_conv_black_white")


main = MainScreen(WIDTH, HEIGHT, FPS, "MNIST_AI", "Iosevka")
canvas = Canvas(main, 28, 28, 15, 5)

predictions = []

def display_table(table):
    distance = 30

    mapped_array = []
    for i in range(len(table)):
        percentage = round(table[i][0] * 100)
        mapped_array.append([i, percentage])

    mapped_array = sorted(mapped_array, key=lambda x: x[1], reverse=True)

    for i in range(len(mapped_array)):
        main.render_text("{} - {}%".format(mapped_array[i][0], mapped_array[i][1]), 610, 110 + (distance * i))


running = True
while running:
    for event in pygame.event.get():
        if (event.type == pygame.KEYDOWN):
            if (event.key == pygame.K_r):
                canvas.clear_screen()
                predictions = []
            if (event.key == pygame.K_s):
                canvas.save()
            if (event.key == pygame.K_t):
                inter = random.randint(0, x_test.shape[0] - 1)
                canvas.set_pixel_array(x_test[inter])

        if (event.type == pygame.QUIT):
            running = False

    main.screen.fill("#181818")
    
    if (not canvas.is_screen_clear()):    
        predictions = predict(network, canvas.get_pixel_array())

    main.render_text("Draw a number:", 20, 20)
    main.render_text("Prediction", 610, 60)
    main.render_button("clear", (20, 615, 100, 30), canvas.clear_screen)
    display_table(predictions)
    
    canvas.render_canvas(20, 50)

    main.update_screen()