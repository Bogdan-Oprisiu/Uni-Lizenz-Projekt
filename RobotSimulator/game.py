import pygame
import math
from controls import RobotController  # existing controller code
from obstacle import generate_obstacle
from robot import Robot

# Extended controller with a method to update using a command string that now supports an optional magnitude.
class ExtendedRobotController(RobotController):
    def update_with_command(self, robot: Robot, command: str) -> None:
        """
        Update the robot based on a command string.

        Supported commands (with optional numeric parameter):
          - "forward [distance]"
          - "backward [distance]"
          - "left [distance]" (strafe left)
          - "right [distance]" (strafe right)
          - "rotate_left [angle]"
          - "rotate_right [angle]"

        If no numeric value is provided, a default of 1.0 is used.
        """
        parts = command.split()
        if not parts:
            return
        cmd = parts[0].lower()
        try:
            magnitude = float(parts[1]) if len(parts) > 1 else 1.0
        except ValueError:
            magnitude = 1.0

        # Compute the robot's current orientation.
        angle_rad = math.radians(robot.angle)
        forward_dx = -math.sin(angle_rad)
        forward_dy = -math.cos(angle_rad)
        right_dx = math.cos(angle_rad)
        right_dy = -math.sin(angle_rad)
        base_speed = self.acceleration_multiplier * robot.acceleration

        if cmd == "forward":
            robot.vx += base_speed * forward_dx * magnitude
            robot.vy += base_speed * forward_dy * magnitude
        elif cmd == "backward":
            robot.vx -= base_speed * forward_dx * magnitude
            robot.vy -= base_speed * forward_dy * magnitude
        elif cmd == "left":
            robot.vx -= base_speed * right_dx * magnitude
            robot.vy -= base_speed * right_dy * magnitude
        elif cmd == "right":
            robot.vx += base_speed * right_dx * magnitude
            robot.vy += base_speed * right_dy * magnitude
        elif cmd == "rotate_left":
            robot.rotate(self.rotation_speed * magnitude)
        elif cmd == "rotate_right":
            robot.rotate(-self.rotation_speed * magnitude)
        # You can add additional commands as needed.

class Game:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 750
        # Create a dummy surface (we won't open an actual window for API-based simulation)
        self.screen = pygame.Surface((self.width, self.height))
        self.robot = Robot(400, 300)
        self.obstacles = [generate_obstacle(self.robot) for _ in range(15)]
        self.controller = ExtendedRobotController(acceleration_multiplier=1.05, rotation_speed=2)
        self.game_over = False

    def reset(self):
        self.robot = Robot(400, 300)
        self.obstacles = [generate_obstacle(self.robot) for _ in range(15)]
        self.game_over = False

    def update(self, command: str):
        if self.game_over:
            return

        self.controller.update_with_command(self.robot, command)
        self.robot.update()

        # Check for collisions using masks.
        robot_mask, robot_offset = self.robot.get_mask()
        collision = False
        for obs in self.obstacles:
            obs_mask, obs_offset = obs.get_mask()
            offset = (obs_offset[0] - robot_offset[0], obs_offset[1] - robot_offset[1])
            if robot_mask.overlap(obs_mask, offset):
                collision = True
                break

        # Check for boundary collisions.
        robot_surface = pygame.Surface((self.robot.width, self.robot.height), pygame.SRCALPHA)
        robot_surface.fill((255, 0, 0))
        rotated_surface = pygame.transform.rotate(robot_surface, self.robot.angle)
        robot_rect = rotated_surface.get_rect(center=(self.robot.x, self.robot.y))
        if (robot_rect.left < 0 or robot_rect.top < 0 or
                robot_rect.right > self.width or robot_rect.bottom > self.height):
            collision = True

        if collision:
            self.game_over = True

    def get_sensor_data(self):
        """
        Compute sensor data based on obstacles relative to the robot's direction.
        Returns a dictionary with keys: "front", "left", "right", "back".
        Each value is the distance to the closest obstacle in that sector,
        or None if no obstacle is detected.
        """
        sensors = {"front": None, "left": None, "right": None, "back": None}
        angle_rad = math.radians(self.robot.angle)
        # Our robot's forward vector.
        forward_dx = -math.sin(angle_rad)
        forward_dy = -math.cos(angle_rad)
        # Compute the robot's forward angle (in degrees).
        robot_forward_angle = math.degrees(math.atan2(forward_dy, forward_dx))
        for obs in self.obstacles:
            dx = obs.x - self.robot.x
            dy = obs.y - self.robot.y
            if dx == 0 and dy == 0:
                continue
            obs_angle = math.degrees(math.atan2(dy, dx))
            # Compute the signed difference in degrees.
            diff = obs_angle - robot_forward_angle
            diff = ((diff + 180) % 360) - 180  # Normalize to (-180, 180]
            distance = math.hypot(dx, dy)
            # Determine sensor sector.
            if abs(diff) <= 45:
                # Front sensor.
                if sensors["front"] is None or distance < sensors["front"]:
                    sensors["front"] = distance
            elif 45 < diff <= 135:
                # Left sensor.
                if sensors["left"] is None or distance < sensors["left"]:
                    sensors["left"] = distance
            elif -135 <= diff < -45:
                # Right sensor.
                if sensors["right"] is None or distance < sensors["right"]:
                    sensors["right"] = distance
            else:
                # Back sensor.
                if sensors["back"] is None or distance < sensors["back"]:
                    sensors["back"] = distance
        return sensors

    def get_state(self):
        return {
            "robot": {
                "x": self.robot.x,
                "y": self.robot.y,
                "angle": self.robot.angle,
                "vx": self.robot.vx,
                "vy": self.robot.vy,
            },
            # "obstacles": [
            #     {"x": obs.x, "y": obs.y, "width": obs.width, "height": obs.height}
            #     for obs in self.obstacles
            # ],
            "game_over": self.game_over,
            "sensors": self.get_sensor_data()
        }
