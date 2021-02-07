#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:09:53 2021

@author: clark
"""

import numpy as np
from numpy import array as arr

test_path = '/home/clark/Computing/python_projects/D-Wave_coding_test/'
test01 = test_path + 'test01.txt'

class Vertex:
    def __init__(self, name = None, spin = None, weight = None, parent = None, children = []):
        self.name = name
        self.spin = spin
        self.weight = weight
        self.parent = parent
        self.children = children
        
    def add_child(self, child_name):
        if str(child_name) not in self.children:
            self.children.append(child_name)


class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight
    


class IsingTree:
    def __init__(self, input_file):
        self.input_file = input_file
        self.remove_comments()
        self.Initialize_VE()
        self.BuildInitialTree()
        self.initial_leaves = self.getLeaves(self.vertices)
        
           
    def remove_comments(self):
        file01 = open(self.input_file)
        contents = file01.read()
        file_lines = contents.split('\n')
        # Pull out the comments and empty lines
        comments = []
        comment_locs = []
        for k in range(len(file_lines)):
            if len(file_lines[k]) == 0:
                comment_locs.append(k)
            else:
                if file_lines[k][0] == 'c':
                    comments.append(file_lines[k])
                    comment_locs.append(k)
        for k  in comment_locs[::-1]:
            file_lines.pop(k)
        self.comments = comments    
        self.file_lines = file_lines
        
    def Initialize_VE(self):
        line0 = self.file_lines[0]
        self.test_info = line0.split()
        num_vertices = int(self.test_info[2])
        self.num_vertices = num_vertices
        vertex_name_list = []
        edge_list = []
        for line in self.file_lines[1:]:
             info = line.split()
             v_info = sorted(info[:2])
             v1 = v_info[0]
             v2 = v_info[1]
             w = int(info[2])
             vertex_name_list.append(v1)
             vertex_name_list.append(v2)
             if str(v1) != str(v2):
                 edge_list.append(Edge(source = v1, destination=v2, weight = w))
                 
                 
        vertex_name_list = sorted(list(set(vertex_name_list)))  
        self.vertex_name_list = vertex_name_list   
        self.edges = edge_list
        vertices = []
        for vname in self.vertex_name_list:
            vertices.append(Vertex(name = vname, weight = 0)) #Now we have all our vertices initialized
        self.vertices = vertices
        assert len(self.vertex_name_list) == self.num_vertices, "Info file gives inconsistent data.  Number of vertices in program line 1 must match number of unique vertices."
        assert len(self.vertices) - len(self.edges) == 1,"The info file does not constitute a tree.  |V|-|E| != 1."
             
        
        
        
    """
    Since we have generic vertices with an attribute 'name'
    we want to look them up by their name
    """
    def getvertex(self, v_list, v_name):
        loc = [v_list[k].name == v_name for k in range(len(self.vertex_name_list))].index(True)
        v = v_list[loc]
        return v  
    """
    def BuildVertices(self):
        vertex_list = []
        for vname in self.vertex_name_list:
            vertex_list.append(Vertex(name = vname, weight = 0)) #Now we have all our vertices initialized
        self.vertex_list = vertex_list
    """    
        
        
        
    def BuildInitialTree(self):
        for line in self.file_lines[1:]: #since the line 0 is the line ['p name ']
            info = line.split()
            v_info = sorted(info[:2])
            parent_name = str(v_info[0])
            child_name = str(v_info[1])
            weight = int(info[2])
            if str(parent_name) == str(child_name):
                """
                Find the vertex with that name
                """
                v = self.getvertex(self.vertices, parent_name)
                v.weight = weight
        
        """
        Now we will get the proper parent/child relationships
        """
        
        children_dict = {}
        for v in self.vertex_name_list:
            children_dict[v] = []
            
        
        for edge in self.edges:
            parent = self.getvertex(self.vertices, edge.source)
            child = self.getvertex(self.vertices, edge.destination)
            child.parent = parent.name
            children_dict[parent.name].append(child.name)
            
        for v in self.vertices:
            v.children = children_dict[v.name]
            
    """
    Let's find all the leaf nodes.  This will be a crucial step
    when we are doing our dynamic programming.
    """
    def getLeaves(self, vertices): #Here we'll pass in a list as we will change the tree
        leaves = []
        for v in vertices:
            if len(v.children) == 0:
                leaves.append(v.name)
        return leaves
        
        
    def starGraphSolver(self, parent_name, leaf_list):
        """
        This particular function is the heart of the mathematical solution.
        In this instance we're takingin a subtree with the structure of a 
        star graph.
        In particular This can be visually represented in two common ways.
        First as a single vertex with many nodes eminating downward/uward/outward
        or second as a central vertex with the leaves in the shape of spokes
        on a bicycle wheel.
        Each subgraph we wish to solve in our program will take this shape.
        Additionally, the explicit minimum energy on a star graph is
        easy to find.  We will take this minimum energy and the spin of the
        parent vertex and wrap this up into a new vertex in a new fractionally
        smaller tree.  Repeating this on all sub star graphs allows us to 
        dynamically capture the explicit minimum energy of the Ising Model/QUBO
        on a tree.
        """
        
        
        
        return
            
        
        

        