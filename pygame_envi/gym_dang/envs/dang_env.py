"""
OpenAI gym environment wrapper
"""
from pygame.locals import *
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import time
import pygame


class DangEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.action_list = ['UP', 'DOWN', 'LEFT','RIGHT']
        self.task = 0
        self.agentX = 5.5
        self.agentY = 8.5
        
    def config(self, param):
        self.params = param  # fault unknow, fix fault by using a tuple: sprite, wall, background loaded from demo
        # self.worldMap =[
        #  [3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2],
        #  [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,2],
        #  [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,2],
        #  [2,2,2,2,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,2,2,2,2,2,2,2,2,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,2,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,2,2,5,5,5,5,5,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #  [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        #  [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        # ]

        self.worldMap = [
            [5,4,4,4,2,2,2,2,4,4],
            [2,0,0,0,0,0,0,4,0,4],
            [5,0,0,0,0,3,3,0,0,4],
            [2,0,2,0,2,3,3,3,0,4],
            [5,0,2,0,2,0,0,0,0,4],
            [2,0,2,0,2,0,0,0,0,4],
            [5,0,2,0,2,0,5,0,2,2],
            [2,0,2,0,0,0,5,0,0,2],
            [5,0,2,0,0,0,5,0,0,2],
            [2,2,2,2,2,2,2,2,2,2]
        ]

        # self.worldMap = [
        #     [5,4,4,4,4,4,2,2,2,2,2,2,2,2,4,4,4,4,4,4],
        #     [5,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,4],
        #     [5,0,0,0,0,0,0,0,0,0,0,0,2,2,3,3,0,0,0,4],
        #     [5,0,0,0,0,2,2,2,3,3,3,3,2,2,3,3,0,0,0,4],
        #     [2,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,4],
        #     [2,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,4],
        #     [2,0,0,2,0,0,0,2,0,0,0,0,0,2,0,2,2,2,2,2],
        #     [2,2,2,2,2,0,0,2,0,0,0,0,0,2,0,0,0,0,0,2],
        #     [2,2,2,2,2,0,0,0,0,0,0,5,2,2,0,0,0,0,0,2],
        #     [2,2,2,2,2,0,0,0,0,0,0,5,2,2,0,0,0,0,0,2],
        #     [2,2,2,2,2,2,2,2,2,0,0,5,2,2,2,2,2,2,2,2],
        #     [2,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,2],
        #     [2,0,0,0,0,0,2,0,0,0,0,5,0,0,0,0,0,0,0,2],
        #     [2,0,0,0,0,0,2,0,0,0,0,5,0,0,4,4,4,0,0,2],
        #     [2,2,2,2,2,2,2,0,0,0,0,5,0,0,4,4,4,0,0,2],
        #     [2,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,2],
        #     [2,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,2,2,2],
        #     [2,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,2,2,2],
        #     [2,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,2],
        #     [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        # ]

        self.sprite_positions=[          
          (1.5, 6.5, 0),
          (7.5, 1.5, 0)          # x y z
        ]
    
        self.t = time.clock() #time of current frame
        self.oldTime = 0. #time of previous frame
        
        self.size = w, h = 640,480
        pygame.init()
        self.window = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Test Env")
        self.screen = pygame.display.get_surface()
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        
        self.f = pygame.font.SysFont(pygame.font.get_default_font(), 20)
        
        self.wm = WorldManager(self.worldMap,self.sprite_positions, self.agentX, self.agentY, -1, 0, 0, 0.6, self.params)
        self.clock.tick(60)
        self.wm.draw(self.screen)
        self.frameTime = float(self.clock.get_time()) / 1000.0 # frameTime is the time this frame has taken, in seconds
        self.t = time.clock()
        self.text = self.f.render(str(self.clock.get_fps()), False, (255, 255, 0))
        self.screen.blit(self.text, self.text.get_rect(), self.text.get_rect())
        pygame.display.flip()

        # speed modifiers
        # self.moveSpeed = self.frameTime * 10.0 # the constant value is in squares / second
        # self.rotSpeed = self.frameTime * 10.0 # the constant value is in radians / second
        self.moveSpeed = 1.0
        self.rotSpeed = np.pi / 2

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.wm.camera.config(self.agentX, self.agentY, -1, 0, 0, 1.0)
        self.draw_screen()
        return self.observation

    @property
    def position(self):
        return round(self.wm.camera.x, 1), round(self.wm.camera.y, 1), int(self.wm.camera.rot)

    def teleport(self, x, y, rot):
        if rot in [0, 90, 180, 270]:
            self.wm.camera.x = x
            self.wm.camera.y = y
            turn_times = (4 - self.wm.camera.rot // 90) + rot // 90
            for i in range(turn_times):
                oldDirX = self.wm.camera.dirx
                self.wm.camera.dirx = self.wm.camera.dirx * math.cos(- self.rotSpeed) - self.wm.camera.diry * math.sin(- self.rotSpeed)
                self.wm.camera.diry = oldDirX * math.sin(- self.rotSpeed) + self.wm.camera.diry * math.cos(- self.rotSpeed)
                oldPlaneX = self.wm.camera.planex
                self.wm.camera.planex = self.wm.camera.planex * math.cos(- self.rotSpeed) - self.wm.camera.planey * math.sin(- self.rotSpeed)
                self.wm.camera.planey = oldPlaneX * math.sin(- self.rotSpeed) + self.wm.camera.planey * math.cos(- self.rotSpeed)

            self.wm.camera.rot = rot

        self.wm.draw(self.screen)
        pygame.display.flip()
        self.draw_screen()
        done = False
        reward = self.reward()
        if reward >= 100: done = True
        # print int(self.wm.camera.x), int(self.wm.camera.y)
        return self.observation, reward, done

    def step(self, action):        
        if action == 0: # move ahead
            moveX = self.wm.camera.x + self.wm.camera.dirx * self.moveSpeed
            if self.worldMap[int(moveX)][int(self.wm.camera.y)]==0:
                self.wm.camera.x = moveX
            moveY = self.wm.camera.y + self.wm.camera.diry * self.moveSpeed
            if self.worldMap[int(self.wm.camera.x)][int(moveY)]==0:
                self.wm.camera.y = moveY

        elif action == 1: # move back
            moveX = self.wm.camera.x - self.wm.camera.dirx * self.moveSpeed
            if(self.worldMap[int(moveX)][int(self.wm.camera.y)]==0 and self.worldMap[int(moveX - 0.1)][int(self.wm.camera.y)]==0):
                self.wm.camera.x = moveX
            moveY = self.wm.camera.y - self.wm.camera.diry * self.moveSpeed
            if(self.worldMap[int(self.wm.camera.x)][int(moveY)]==0 and self.worldMap[int(self.wm.camera.x)][int(moveY - 0.1)]==0):
                self.wm.camera.y = moveY
        elif action == 3: # turn right
            oldDirX = self.wm.camera.dirx
            self.wm.camera.dirx = self.wm.camera.dirx * math.cos(- self.rotSpeed) - self.wm.camera.diry * math.sin(- self.rotSpeed)
            self.wm.camera.diry = oldDirX * math.sin(- self.rotSpeed) + self.wm.camera.diry * math.cos(- self.rotSpeed)
            oldPlaneX = self.wm.camera.planex
            self.wm.camera.planex = self.wm.camera.planex * math.cos(- self.rotSpeed) - self.wm.camera.planey * math.sin(- self.rotSpeed)
            self.wm.camera.planey = oldPlaneX * math.sin(- self.rotSpeed) + self.wm.camera.planey * math.cos(- self.rotSpeed)
            self.wm.camera.rot += 90
            if self.wm.camera.rot == 360:
                self.wm.camera.rot = 0
        elif action == 2: # turn left
            oldDirX = self.wm.camera.dirx
            self.wm.camera.dirx = self.wm.camera.dirx * math.cos(self.rotSpeed) - self.wm.camera.diry * math.sin(self.rotSpeed)
            self.wm.camera.diry = oldDirX * math.sin(self.rotSpeed) + self.wm.camera.diry * math.cos(self.rotSpeed)
            oldPlaneX = self.wm.camera.planex
            self.wm.camera.planex = self.wm.camera.planex * math.cos(self.rotSpeed) - self.wm.camera.planey * math.sin(self.rotSpeed)
            self.wm.camera.planey = oldPlaneX * math.sin(self.rotSpeed) + self.wm.camera.planey * math.cos(self.rotSpeed)
            self.wm.camera.rot -= 90
            if self.wm.camera.rot < 0:
                self.wm.camera.rot = 270

        self.wm.draw(self.screen)
        pygame.display.flip()
        self.draw_screen()
        done = False
        reward = self.reward()
        if reward >= 100: done = True
        # print int(self.wm.camera.x), int(self.wm.camera.y)
        return self.observation, reward, done

    def reward(self):
        print("info", self.wm.object_seen)
        if self.wm.object_seen == False: return -1 
        reward = -1

        pos = self.sprite_positions[self.task]
        xx = pos[0] - (self.wm.camera.x)
        yy = pos[1] - (self.wm.camera.y)
        print(math.sqrt(xx*xx+yy*yy))
        if math.sqrt(xx*xx+yy*yy) <= 1.0 :
            reward = 10
    
        return reward

    def _render(self, mode='human', close=False):
        ### TODO render map
        return True

    def action_list(self):
        return self.action_list

    def draw_screen(self):
        img = pygame.surfarray.array3d(self.screen)
        self.observation = img.swapaxes(0, 1)
        return self.observation

    def get_action_list(self):
        return self.action_list

    def config_task(self, number = 0):
        self.task = number

    def config_agent_position(self, x = 5, y = 5):
        self.agentY = y
        self.agentX = x

class WorldManager(object):

    def __init__(self,worldMap,sprite_positions,x,y,dirx,diry,planex,planey, params):
        self.params = params
        self.sprites = params[0]
        
        self.background = params[2]
        self.images = params[1]
        self.camera = Camera(x,y,dirx,diry,planex,planey)
        self.worldMap = worldMap
        self.sprite_positions = sprite_positions
        self.object_seen = True
        
    def draw(self, surface):
        
        texWidth = 64
        texHeight = 64
        w = surface.get_width()
        h = surface.get_height()
        #draw background
        if self.background is None:
            self.background = self.params[2]
        surface.blit(self.background, (0,0))
        zBuffer = []
        for x in range(w):
            #calculate ray position and direction 
            cameraX = float(2 * x / float(w) - 1) #x-coordinate in camera space
            rayPosX = self.camera.x
            rayPosY = self.camera.y
            rayDirX = self.camera.dirx + self.camera.planex * cameraX
            rayDirY = self.camera.diry + self.camera.planey * cameraX
            #which box of the map we're in  
            mapX = int(rayPosX)
            mapY = int(rayPosY)
                
            #length of ray from current position to next x or y-side
            sideDistX = 0.
            sideDistY = 0.
       
            #length of ray from one x or y-side to next x or y-side
            deltaDistX = math.sqrt(1 + (rayDirY * rayDirY) / (rayDirX * rayDirX))
            if rayDirY == 0: rayDirY = 0.00001
            deltaDistY = math.sqrt(1 + (rayDirX * rayDirX) / (rayDirY * rayDirY))
            perpWallDist = 0.
       
            #what direction to step in x or y-direction (either +1 or -1)
            stepX = 0
            stepY = 0

            hit = 0 # was there a wall hit?
            side = 0 # was a NS or a EW wall hit?
            
            # calculate step and initial sideDist
            if rayDirX < 0:
                stepX = - 1
                sideDistX = (rayPosX - mapX) * deltaDistX
            else:
                stepX = 1
                sideDistX = (mapX + 1.0 - rayPosX) * deltaDistX
                
            if rayDirY < 0:
                stepY = - 1
                sideDistY = (rayPosY - mapY) * deltaDistY
            else:
                stepY = 1
                sideDistY = (mapY + 1.0 - rayPosY) * deltaDistY
                
            # perform DDA
            while hit == 0:
                # jump to next map square, OR in x - direction, OR in y - direction
                if sideDistX < sideDistY:
        
                    sideDistX += deltaDistX
                    mapX += stepX
                    side = 0
                else:
                    sideDistY += deltaDistY
                    mapY += stepY
                    side = 1

                # Check if ray has hit a wall
                if (self.worldMap[mapX][mapY] > 0): 
                    hit = 1
                
            # Calculate distance projected on camera direction (oblique distance will give fisheye effect !)
            if (side == 0):
                #perpWallDist = fabs((mapX - rayPosX + (1 - stepX) / 2) / rayDirX)
                perpWallDist = (abs((mapX - rayPosX + (1 - stepX) / 2) / rayDirX))
            else:
                perpWallDist = (abs((mapY - rayPosY + (1 - stepY) / 2) / rayDirY))
      
            # Calculate height of line to draw on surface
            if perpWallDist == 0:perpWallDist = 0.000001
            lineHeight = abs(int(h / perpWallDist))
       
            # calculate lowest and highest pixel to fill in current stripe
            drawStart = - lineHeight / 2 + h / 2
            drawEnd = lineHeight / 2 + h / 2
        
            #texturing calculations
            texNum = self.worldMap[mapX][mapY] - 1; #1 subtracted from it so that texture 0 can be used!
           
            #calculate value of wallX
            wallX = 0 #where exactly the wall was hit
            if (side == 1):
                wallX = rayPosX + ((mapY - rayPosY + (1 - stepY) / 2) / rayDirY) * rayDirX
            else:
                wallX = rayPosY + ((mapX - rayPosX + (1 - stepX) / 2) / rayDirX) * rayDirY;
            wallX -= math.floor((wallX));
           
            #x coordinate on the texture
            texX = int(wallX * float(texWidth))
            if(side == 0 and rayDirX > 0): 
                texX = texWidth - texX - 1;
            if(side == 1 and rayDirY < 0): 
                texX = texWidth - texX - 1;

            if(side == 1):
                texNum +=8
            if lineHeight > 10000:
                lineHeight=10000
                drawStart = -10000 //2 + h//2
            surface.blit(pygame.transform.scale(self.images[texNum][texX], (1, lineHeight)), (x, drawStart))
            zBuffer.append(perpWallDist)


        def sprite_compare(s1):
            import math
            s1Dist = math.sqrt((s1[0] -self.camera.x) ** 2 + (s1[1] -self.camera.y) ** 2)
            return s1Dist
            # s2Dist = math.sqrt((s2[0] -self.camera.x) ** 2 + (s2[1] -self.camera.y) ** 2)  
            # if s1Dist>s2Dist:
            #     return -1
            # elif s1Dist==s2Dist:
            #     return 0
            # else:
            #     return 1
        #draw sprites
        
        self.sprite_positions.sort(key=sprite_compare)
        self.object_seen = False
        for sprite in self.sprite_positions:
            #translate sprite position to relative to camera
            spriteX = sprite[0] - self.camera.x;
            spriteY = sprite[1] - self.camera.y;
             
            #transform sprite with the inverse camera matrix
            # [ self.camera.planex   self.camera.dirx ] -1                                       [ self.camera.diry      -self.camera.dirx ]
            # [               ]       =  1/(self.camera.planex*self.camera.diry-self.camera.dirx*self.camera.planey) *   [                 ]
            # [ self.camera.planey   self.camera.diry ]                                          [ -self.camera.planey  self.camera.planex ]
          
            invDet = 1.0 / (self.camera.planex * self.camera.diry - self.camera.dirx * self.camera.planey) #required for correct matrix multiplication
          
            transformX = invDet * (self.camera.diry * spriteX - self.camera.dirx * spriteY)
            transformY = invDet * (-self.camera.planey * spriteX + self.camera.planex * spriteY) #this is actually the depth inside the surface, that what Z is in 3D       
            
            if transformY == 0:
                return 

            spritesurfaceX = int((w / 2) * (1 + transformX / transformY))
          
            #calculate height of the sprite on surface
            spriteHeight = abs(int(h / (transformY))) #using "transformY" instead of the real distance prevents fisheye
            #calculate lowest and highest pixel to fill in current stripe
            drawStartY = -spriteHeight // 2 + h // 2
            drawEndY = spriteHeight // 2 + h // 2
          
            #calculate width of the sprite
            spriteWidth = abs( int (h / (transformY)))
            drawStartX = -spriteWidth // 2 + spritesurfaceX
            drawEndX = spriteWidth // 2 + spritesurfaceX
            if spriteHeight < 1000:
                for stripe in range(int(drawStartX), int(drawEndX)):
                    texX = int(256 * (stripe - (-spriteWidth / 2 + spritesurfaceX)) * texWidth / spriteWidth) // 256
                    #the conditions in the if are:
                    ##1) it's in front of camera plane so you don't see things behind you
                    ##2) it's on the surface (left)
                    ##3) it's on the surface (right)
                    ##4) ZBuffer, with perpendicular distance
                    if(transformY > 0 and stripe > 0 and stripe < w and transformY < zBuffer[stripe]):
                        self.object_seen = True
                        surface.blit(pygame.transform.scale(self.sprites[sprite[2]][texX], (1, spriteHeight)), (stripe, drawStartY))

class Camera(object):
    def __init__(self,x,y,dirx,diry,planex,planey):
        self.x = float(x)
        self.y = float(y)
        self.rot = 0
        self.dirx = float(dirx)
        self.diry = float(diry)
        self.planex = float(planex)
        self.planey = float(planey)
    
    def config(self, x,y,dirx,diry,planex,planey):
        self.x = float(x)
        self.y = float(y)
        self.dirx = float(dirx)
        self.diry = float(diry)
        self.planex = float(planex)
        self.planey = float(planey)

    def get_info(self):
        return self.x, self.y, self.dirx, self.diry, self.planex, self.planey


def convert_to_grid(camera):
    # convert position
    # convert direction
    return new_camera