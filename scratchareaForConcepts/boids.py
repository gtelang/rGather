import pylab
import copy 
from visual import *
from random import randrange
from functools import partial
import time

from matplotlib import pyplot as pot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Boids:
    def __init__(self, numboids ):     #class constructor with default parameters filled
        #class constants 
        display(title = "Boids v1.0")   #put a title in the display window

               #the next six lines define the boundaries of the torus



        self.RADIUS = 5              #radius of a boid.  I wimped and used spheres.
        self.NEARBY =  10   #the 'halo' of space around each boid
        self.FACTOR = .95           #the amount of movement to the perceived flock center
        self.NEGFACTOR = self.FACTOR * -1.0 #same thing, only negative

        self.boidflock_velocity = []    #empty list of boids for velocity

        self.NUMBOIDS = numboids        #the number of boids in the flock

        self.boidflock = []             #empty list of boids
        self.DT = 0.01                #delay time between snapshots



        global postion_collection_boids  

        self.boids()                    #okay, now that all the constants have initialized, let's fly!



    def boids(self):
        global postion_collection_boids
        global cu_pos
        global pos_append
        self.initializePositions()      #create a space with boids
        simulation_time = 100
        current_time = 0

        while (current_time <= simulation_time):                   #loop forever

            rate(100)                   #controls the animation speed, bigger = faster
            a = self.moveAllBoidsToNewPositions()   #um ... what it says


            cu_pos = copy(a)

            pos_append()


            current_time = current_time + self.DT

    def initializePositions(self):
                #splatter a flock in the space randomly
        c = 0                                   #initialize the color switch
        for b in range(self.NUMBOIDS):          #for each boid, ...
            x = randrange(20, 200) #random left-right
            y = randrange(20, 200) #random up-down
            z = randrange(20, 200) #random front-back

##            x = randrange(8, 10) #random left-right # done by me 
##            y = randrange(8, 10) #random up-down
##            z = randrange(8, 10) #random front-back



            if c > 4:                           #reset the color switch when it grows too big
                c = 0
            if c == 0:
                COLOR = color.yellow            #a third of the boids shall have yellow
            if c == 1:
                COLOR = color.red               #and yea a third of the boids shall have red
            if c == 2:
                COLOR = color.blue              #and verily a third of the boids shall have blue
            if c==3: 
                COLOR = color.green
            if c==4: 
                COLOR = color.white


            #splat a boid, add to flock list
            self.boidflock.append(sphere(pos=(x,y,z), radius=self.RADIUS, color=COLOR))

            c = c + 1                           #increment the color switch



            vx = 0
            vy = 0 
            vz = 0

            self.vel = vector(vx,vy,vz)

            self.boidflock_velocity.append(self.vel)



    def moveAllBoidsToNewPositions(self):
        pos_collec_boids = []

        for b in range(self.NUMBOIDS):



            v1 = vector(0.0,0.0,0.0)        #initialize vector for rule 1
            v2 = vector(0.0,0.0,0.0)        #initialize vector for rule 2
            v3 = vector(0.0,0.0,0.0)        #initialize vector for rule 3


            v1 = self.rule1(b)              #get the vector for rule 1
            v2 = self.rule2(b)              #get the vector for rule 2
            v3 = self.rule3(b)              #get the vector for rule 3

            boidvelocity = vector(0.0,0.0,0.0)          #initialize the boid velocity
            boidvelocity = boidvelocity + v1 + v2 + v3  #accumulate the rules vector results


            # limiting the boid veloicity 

            vlimit = 10

            bvelocity =  boidvelocity*self.DT

            if (mag(bvelocity))> vlimit : 

                    bvelocity = norm(bvelocity)*vlimit


            self.boidflock[b].pos = self.boidflock[b].pos + (bvelocity) #move the boid
            bvelocity =  boidvelocity*self.DT
            self.boidflock_velocity[b] = bvelocity



            pos_collec_boids.append(self.boidflock[b].pos)
        return  pos_collec_boids





    def rule1(self, aboid):    #Rule 1:  boids fly to perceived flock center

        pfc = vector(0.0,0.0,0.0)                   #pfc: perceived flock center

        for b in range(self.NUMBOIDS):              #for all the boids
            if b != aboid:                          #except the boid at hand
                 pfc = pfc + self.boidflock[b].pos  #calculate the total pfc

        pfc = pfc/(self.NUMBOIDS - 1.0)             #average the pfc


        #nudge the boid in the correct direction toward the pfc 
        if pfc.x > self.boidflock[aboid].x:
             pfc.x = (pfc.x - self.boidflock[aboid].x)*self.FACTOR
        if pfc.x < self.boidflock[aboid].x:
             pfc.x = (self.boidflock[aboid].x - pfc.x)*self.NEGFACTOR
        if pfc.y > self.boidflock[aboid].y:
             pfc.y = (pfc.y - self.boidflock[aboid].y)*self.FACTOR
        if pfc.y < self.boidflock[aboid].y:
             pfc.y = (self.boidflock[aboid].y - pfc.y)*self.NEGFACTOR
        if pfc.z > self.boidflock[aboid].z:
             pfc.z = (pfc.z - self.boidflock[aboid].z)*self.FACTOR
        if pfc.z < self.boidflock[aboid].z:
             pfc.z = (self.boidflock[aboid].z - pfc.z)*self.NEGFACTOR



        return pfc 

    def rule2(self, aboid):    #Rule 2: boids avoid other boids
        v = vector(0.0,0.0,0.0) #initialize the avoidance vector

        for b in range(self.NUMBOIDS):
            if b != aboid:
                if abs(self.boidflock[b].x - self.boidflock[aboid].x) < self.NEARBY:
                    if self.boidflock[b].x > self.boidflock[aboid].x:
                        v.x = self.NEARBY * 12.0    #works better when I multiply by 12, don't know why
                    if self.boidflock[b].x < self.boidflock[aboid].x:
                        v.x = -self.NEARBY * 12.0
                if abs(self.boidflock[b].y - self.boidflock[aboid].y) < self.NEARBY:
                    if self.boidflock[b].y > self.boidflock[aboid].y:
                        v.y = self.NEARBY * 12.0
                    if self.boidflock[b].y < self.boidflock[aboid].y:
                        v.y = -self.NEARBY * 12.0
                if abs(self.boidflock[b].z - self.boidflock[aboid].z) < self.NEARBY:
                    if self.boidflock[b].z > self.boidflock[aboid].z:
                        v.z = self.NEARBY * 12.0
                    if self.boidflock[b].z < self.boidflock[aboid].z:
                        v.z = -self.NEARBY * 12.0


        return v

    def rule3(self, aboid):    #Rule 3: boids try to match speed of flock
        pfv = vector(0.0,0.0,0.0)   #pfv: perceived flock velocity

        for b in range(self.NUMBOIDS):
            if b != aboid:
                 pfv = pfv + self.boidflock_velocity[b]

        pfv = pfv/(self.NUMBOIDS - 1.0)
        pfv = pfv/(aboid + 1)    


        return pfv


if __name__ == "__main__":
    #if you run this from Idle via the F5 key, the following occurs:
    number_of_members = 20




    cu_pos = 0
    postion_collection_boids  = [] 

    dt = 1.0

    def pos_append():
        global cu_pos
        global postion_collection_boids


        postion_collection_boids.append(cu_pos)


        #print " i am herenjadhfjklahlk"




    b = Boids(number_of_members)     #instantiate the Boids class, the class constructor takes care of the rest.

##    b = Boids(20, 60.0)   #here's a way to change the flock amount, and space size




pos_coll_boids =  postion_collection_boids


# function for  calculaiton of individual postion from the nodes 
def postion_extraction_boids(no_boids,colection_boids_position): 

    individual_boids_pos_collection = [] # collection of individual position of all boids 
    individual_boids_pos = [] # individual position only

    i = 0

    while i < no_boids: 
        j = 0

        while j < len(colection_boids_position):

            pos1 = colection_boids_position[j][i]
            individual_boids_pos.append(pos1)
            j = j+1
        individual_boids_pos_collection.append(individual_boids_pos)
        individual_boids_pos = []
        i = i +1
    return individual_boids_pos_collection



def velocity_calculation(collection_of_individual_pos,time_gap_between_each_position):

    i = 0
    ind_velocity = []
    ind_velocity_collecction = []

    while i < len(collection_of_individual_pos):

        j = 0 
        initial_position = collection_of_individual_pos[i][j]


        while (j < len(collection_of_individual_pos[i]) - 1) : # so that j should not exceed size of array


            j = j+1
            second_position = collection_of_individual_pos[i][j]

            x1 = initial_position[0]
            y1 = initial_position[1]
            z1 = initial_position[2]



            x2 = second_position[0]
            y2 = second_position[1]
            z2 = second_position[2]

            velocity1 = math.sqrt((math.pow((x2-x1),2)) + (math.pow((y2-y1),2)) + (math.pow((z2-z1),2)))

            ind_velocity.append(velocity1)

            initial_position = second_position
        ind_velocity_collecction.append(ind_velocity)
        ind_velocity =[]
        i = i + 1
    return ind_velocity_collecction 




final_position_collection = postion_extraction_boids(number_of_members,pos_coll_boids)


print 'i should be 5',type(final_position_collection)
print 'i should be 200',type(final_position_collection[0])
print 'i should be 3',type(final_position_collection[0][0])


final_velcoity_collection = velocity_calculation(final_position_collection,dt)


print "yes i can do something here"


#******************************************************************************
# 3d ploting code here 

# Here all the atoms are stacked into a (4, 3) array
atoms = np.vstack([final_position_collection[0][:]])
atoms1 = np.vstack([final_position_collection[1][:]])
atoms2 = np.vstack([final_position_collection[2][:]])
atoms3 = np.vstack([final_position_collection[3][:]])



ax = plt.subplot(111, projection='3d')

# Plot scatter of points
m = 0
n = 9000


# two different kind of plots just to give connection effect of the graph
ax.scatter3D(atoms[m:n, 0], atoms[m:n, 1], atoms[m:n, 2] )
ax.plot(atoms[m:n, 0], atoms[m:n, 1], atoms[m:n, 2])

ax.scatter3D(atoms1[m:n, 0], atoms1[m:n, 1], atoms1[m:n, 2] )
ax.plot(atoms1[m:n, 0], atoms1[m:n, 1], atoms1[m:n, 2])

ax.scatter3D(atoms2[m:n, 0], atoms2[m:n, 1], atoms2[m:n, 2] )
ax.plot(atoms2[m:n, 0], atoms2[m:n, 1], atoms2[m:n, 2])

ax.scatter3D(atoms3[m:n, 0], atoms3[m:n, 1], atoms3[m:n, 2] )
ax.plot(atoms3[m:n, 0], atoms3[m:n, 1], atoms3[m:n, 2])




plt.show()
