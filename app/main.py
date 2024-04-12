import pygame
import time
import math
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
        self.pixel_array = ["#525252" for _ in range(self.column * self.row)]

    def render_canvas(self, offset_x, offset_y):
        for y in range(self.row):
            for x in range(self.column):
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_down = pygame.mouse.get_pressed()[0]

                pos_x = x * (self.size + self.padding) + offset_x
                pos_y = y * (self.size + self.padding) + offset_y

                distance = math.sqrt(math.pow(mouse_x - pos_x, 2) + math.pow(mouse_y - pos_y, 2))
                if (distance < 25 and mouse_down):
                    self.pixel_array[y * self.row + x] = "#ffffff"

                pygame.draw.rect(self.screen.screen, self.pixel_array[y * self.row + x], pygame.Rect(pos_x, pos_y, self.size, self.size), 0, 2)


WIDTH = 800
HEIGHT = 700
FPS = 60


main = MainScreen(WIDTH, HEIGHT, FPS, "MNIST_AI", "Iosevka")
canvas = Canvas(main, 28, 28, 15, 5)


running = True

while running:
    for event in pygame.event.get():
        if (event.type == pygame.KEYDOWN):
            if (event.key == pygame.K_r):
                canvas.clear_screen()
        if (event.type == pygame.QUIT):
            running = False

    main.screen.fill("#181818")

    
    main.render_text("Draw a number:", 20, 20)
    main.render_text("Prediction", 650, 60)
    main.render_button("clear", (20, 615, 100, 30), canvas.clear_screen)
    
    canvas.render_canvas(20, 50)

    main.update_screen()