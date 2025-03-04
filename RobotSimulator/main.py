import math

import pygame

from controls import RobotController  # Import the controller class
from obstacle import generate_obstacle
from robot import Robot

pygame.init()
screen = pygame.display.set_mode((1200, 750))
pygame.display.set_caption("Robot Simulator")
clock = pygame.time.Clock()

# Create an instance of the robot (with default width, height, angle, and physics properties)
robot = Robot(400, 300)
# Generate a list of obstacles (e.g., 15 obstacles)
obstacles = [generate_obstacle(robot) for _ in range(15)]
# Create a RobotController instance for handling controls
controller = RobotController(acceleration_multiplier=1.05, rotation_speed=2)

# Initialize score and a font to display it.
score = 0.0  # score as a float representing seconds survived while moving
score_font = pygame.font.Font(None, 36)  # smaller font for in-game display
movement_threshold = 0.1  # Only count movement if velocity is above this value

running = True
while running:
    # dt is the time in seconds for this frame.
    dt = clock.tick(60) / 1000  # 60 frames per second

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Save the robot's previous state (position, velocity, and angle)
    old_x, old_y, old_angle = robot.x, robot.y, robot.angle
    if hasattr(robot, 'vx') and hasattr(robot, 'vy'):
        old_vx, old_vy = robot.vx, robot.vy

    # Update robot state using the controller and physics update
    controller.update(robot)
    robot.update()

    # Increase score only if the robot is moving (velocity magnitude above threshold)
    velocity = math.hypot(robot.vx, robot.vy)
    if velocity > movement_threshold:
        score += dt

    # Collision detection using masks
    robot_mask, robot_offset = robot.get_mask()
    collision = False
    for obs in obstacles:
        obs_mask, obs_offset = obs.get_mask()
        offset = (obs_offset[0] - robot_offset[0], obs_offset[1] - robot_offset[1])
        if robot_mask.overlap(obs_mask, offset):
            collision = True
            break

    # Boundary collision check:
    robot_surface = pygame.Surface((robot.width, robot.height), pygame.SRCALPHA)
    robot_surface.fill((255, 0, 0))
    rotated_surface = pygame.transform.rotate(robot_surface, robot.angle)
    robot_rect = rotated_surface.get_rect(center=(robot.x, robot.y))
    screen_width, screen_height = screen.get_size()
    if (robot_rect.left < 0 or robot_rect.top < 0 or
            robot_rect.right > screen_width or robot_rect.bottom > screen_height):
        collision = True

    if collision:
        # Display "Game Over" screen with final score
        font = pygame.font.Font(None, 74)
        game_over_text = font.render("Game Over", True, (255, 255, 255))
        game_over_rect = game_over_text.get_rect(center=(screen_width // 2, screen_height // 2 - 50))
        score_text = font.render(f"Score: {int(score)}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(screen_width // 2, screen_height // 2 + 50))
        screen.fill((0, 0, 0))
        screen.blit(game_over_text, game_over_rect)
        screen.blit(score_text, score_rect)
        pygame.display.flip()
        pygame.time.delay(2000)  # Display message for 2 seconds
        running = False
        continue

    # Draw everything
    screen.fill((30, 30, 30))  # Clear the screen with a dark gray background
    for obs in obstacles:
        obs.draw(screen)
    robot.draw(screen)

    # Draw the score in the bottom right corner
    score_disp = score_font.render(f"Score: {int(score)}", True, (255, 255, 255))
    score_rect = score_disp.get_rect(bottomright=(screen_width - 10, screen_height - 10))
    screen.blit(score_disp, score_rect)

    pygame.display.flip()

pygame.quit()
