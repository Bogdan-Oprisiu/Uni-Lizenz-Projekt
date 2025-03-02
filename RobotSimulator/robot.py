import pygame

class Robot:
    def __init__(self, x, y, width=30, height=35, angle=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle  # Angle in degrees

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self, d_angle):
        self.angle += d_angle

    def draw(self, screen):
        # Create a surface for the robot
        robot_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        robot_surface.fill((255, 0, 0))  # Fill it red
        # Rotate the robot surface
        rotated_surface = pygame.transform.rotate(robot_surface, self.angle)
        rect = rotated_surface.get_rect(center=(self.x, self.y))
        screen.blit(rotated_surface, rect)

    def get_mask(self):
        """
        Create and return a mask for the rotated robot along with the top-left position of the mask.
        """
        robot_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        robot_surface.fill((255, 0, 0))
        rotated_surface = pygame.transform.rotate(robot_surface, self.angle)
        mask = pygame.mask.from_surface(rotated_surface)
        rect = rotated_surface.get_rect(center=(self.x, self.y))
        return mask, rect.topleft
