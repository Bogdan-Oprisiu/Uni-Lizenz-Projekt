import pygame
import math

class RobotController:
    def __init__(self, acceleration_multiplier=1.0, rotation_speed=5):
        self.acceleration_multiplier = acceleration_multiplier  # This will be multiplied by robot.acceleration
        self.rotation_speed = rotation_speed

    def update(self, robot):
        """
        Updates the robot's velocity and rotation based on keyboard input.
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
            # Accelerate forward relative to robot's orientation
            robot.vx += self.acceleration_multiplier * robot.acceleration * forward_dx
            robot.vy += self.acceleration_multiplier * robot.acceleration * forward_dy
        if keys[pygame.K_s]:
            # Accelerate backward (negative forward)
            robot.vx -= self.acceleration_multiplier * robot.acceleration * forward_dx
            robot.vy -= self.acceleration_multiplier * robot.acceleration * forward_dy
        if keys[pygame.K_a]:
            # Accelerate left (strafe left, negative right vector)
            robot.vx -= self.acceleration_multiplier * robot.acceleration * right_dx
            robot.vy -= self.acceleration_multiplier * robot.acceleration * right_dy
        if keys[pygame.K_d]:
            # Accelerate right (strafe right)
            robot.vx += self.acceleration_multiplier * robot.acceleration * right_dx
            robot.vy += self.acceleration_multiplier * robot.acceleration * right_dy
        if keys[pygame.K_q]:
            robot.rotate(self.rotation_speed)
        if keys[pygame.K_e]:
            robot.rotate(-self.rotation_speed)
