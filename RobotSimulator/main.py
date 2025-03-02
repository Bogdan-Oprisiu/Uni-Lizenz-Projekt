import pygame
import math
from robot import Robot
from obstacle import generate_obstacle
from controls import RobotController  # Import the controller class

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1200, 750))
pygame.display.set_caption("Robot Simulator")
clock = pygame.time.Clock()

# Create an instance of the robot (with default width, height, angle)
robot = Robot(400, 300)
# Generate a list of obstacles (e.g., 15 obstacles)
obstacles = [generate_obstacle(robot) for _ in range(15)]

# Create a RobotController instance for handling controls
controller = RobotController(speed=5, rotation_speed=5)

running = True
while running:
    clock.tick(60)  # 60 frames per second
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Save the robot's previous state (position and angle)
    old_x, old_y, old_angle = robot.x, robot.y, robot.angle

    # Update robot state using the controller
    controller.update(robot)

    # Collision detection with obstacles using masks
    robot_mask, robot_offset = robot.get_mask()
    collision = False
    for obs in obstacles:
        obs_mask, obs_offset = obs.get_mask()
        # Calculate the offset between the robot and the obstacle
        offset = (obs_offset[0] - robot_offset[0], obs_offset[1] - robot_offset[1])
        if robot_mask.overlap(obs_mask, offset):
            collision = True
            break

    # Boundary collision check:
    # Recalculate the robot's rotated surface and get its bounding rect.
    robot_surface = pygame.Surface((robot.width, robot.height), pygame.SRCALPHA)
    robot_surface.fill((255, 0, 0))
    rotated_surface = pygame.transform.rotate(robot_surface, robot.angle)
    robot_rect = rotated_surface.get_rect(center=(robot.x, robot.y))
    screen_width, screen_height = screen.get_size()

    # If any part of the robot is outside the window, treat it as a collision.
    if robot_rect.left < 0 or robot_rect.top < 0 or robot_rect.right > screen_width or robot_rect.bottom > screen_height:
        collision = True

    # If a collision occurs (either with obstacles or screen borders), revert to the previous state
    if collision:
        robot.x, robot.y, robot.angle = old_x, old_y, old_angle

    # Draw everything
    screen.fill((30, 30, 30))  # Clear the screen with a dark gray background
    for obs in obstacles:
        obs.draw(screen)
    robot.draw(screen)
    pygame.display.flip()

pygame.quit()
