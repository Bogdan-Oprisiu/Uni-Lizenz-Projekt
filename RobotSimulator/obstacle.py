import pygame
import math
import random

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.y, self.width, self.height))

    def get_mask(self):
        # Create a surface representing the obstacle
        obstacle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(obstacle_surface, (255, 255, 255), (0, 0, self.width, self.height))
        mask = pygame.mask.from_surface(obstacle_surface)
        # Return the mask along with the obstacle's top-left corner as the offset
        return mask, (self.x, self.y)

def generate_obstacle(robot, min_distance=100, screen_width=800, screen_height=600):
    """
    Generate an obstacle at a random position and size,
    ensuring that the obstacle is at least `min_distance` away from the robot's center.
    """
    while True:
        width = random.randint(10, 100)
        height = random.randint(10, 100)
        x = random.randint(0, screen_width - width)
        y = random.randint(0, screen_height - height)
        # Calculate the center of the obstacle
        obstacle_center_x = x + width / 2
        obstacle_center_y = y + height / 2
        # Compute Euclidean distance from the robot's center
        dx = obstacle_center_x - robot.x
        dy = obstacle_center_y - robot.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance >= min_distance:
            return Obstacle(x, y, width, height)
