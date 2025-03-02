import pygame
import math


class RobotController:
    def __init__(self, speed=5, rotation_speed=5):
        self.speed = speed
        self.rotation_speed = rotation_speed

    def update(self, robot):
        """
        Updates the robot's position and rotation based on keyboard input.
        Movement is computed relative to the robot's current orientation.
        """
        # Calculate movement vectors relative to the robot's orientation.
        angle_rad = math.radians(robot.angle)
        # Forward vector (assuming angle 0 means facing up)
        forward_dx = -math.sin(angle_rad)
        forward_dy = -math.cos(angle_rad)
        # Right vector (perpendicular to forward)
        right_dx = math.cos(angle_rad)
        right_dy = -math.sin(angle_rad)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            robot.move(self.speed * forward_dx, self.speed * forward_dy)
        if keys[pygame.K_s]:
            robot.move(-self.speed * forward_dx, -self.speed * forward_dy)
        if keys[pygame.K_a]:
            robot.move(-self.speed * right_dx, -self.speed * right_dy)
        if keys[pygame.K_d]:
            robot.move(self.speed * right_dx, self.speed * right_dy)
        if keys[pygame.K_q]:
            robot.rotate(self.rotation_speed)
        if keys[pygame.K_e]:
            robot.rotate(-self.rotation_speed)
