# CS3243-Projects
A repository of my solutions for CS3243 - Introduction to Artificial Intelligence project assignments. Passes all test cases on codePost (for now).

## Project 1
Project 1's task is to apply classic search algorithms (BFS, DFS, UCS/Dijkstra's, A*) on a fun chess puzzle: the King's Maze. 
The algorithm's implementation is pretty straightforward, you can find a million articles on each of them just by Googling and reading up AIMA 4th edition.
The difficult part was parsing the test cases then setting up the data structures, and wrestling with the Python 
(I'm most comfortable with C-like statically-typed languages; programming without types scares me).

**Thoughts:**

While solving `BFS.py`, I fell into the OOP trap and hardly made any progress for a couple of days before I eventually got it to work. 
I was too worried about imaginary problems such as "code maintainability" and "code extensibility" that it distracted me from the main task: 
implement the damn algorithm! For `DFS.py`, I just copied my solution from BFS and changed the frontier from a queue to a stack. Since I value my time, I could 
not bother with OOP anymore, proceeded to break every principle and made everything a global state (sorry Uncle Bob), and managed to solve UCS and A* 
– which included debugging with cryptid feedback on codePost within 2 hours, hence this should explain why there's such a difference in the programming
styles between the Python files.

## Project 2
TODO

## Disclaimer
I did not come up with the projects and tasks - they are the result of the hard work by the staff supporting and teaching this module.

## Warning
My Python skills are pretty cringe at the moment, so please bear with all the paranoid type annotations.
