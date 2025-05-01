# nnodely Applications
Applications of Model-Structured Neural Networks using nnodely.
This repository contains a list of applications of the nnodely framework with relative reference.

## General Applications
### Mass-Spring-Damper System
The file presents the modeling of a mass-spring-damper system.
The network estimate the next position and the next velocity of the mass.

### Pendulum
The file presents the modeling of a pendulum.
The network estimate the next position of the pendulum.

### Nonlinear Function Fitting
The file presents the modeling of a nonlinear function.
The network estimate the value of the function with a family of models.

## Vehicle Applications
### Longitudinal Vehicle Dynamics
The file presents the modeling of the longitudinal vehicle dynamics presented in:

    @article{DaLio2020Modelling,
        author = {Mauro Da Lio, Daniele Bortoluzzi and Gastone Pietro Rosati Papini},
        title = {Modelling longitudinal vehicle dynamics with neural networks},
        journal = {Vehicle System Dynamics},
        volume = {58},
        number = {11},
        pages = {1675--1693},
        year = {2020},
        publisher = {Taylor \& Francis},
        doi = {10.1080/00423114.2019.1638947}
    }

### Lateral Vehicle Dynamics
The file presents the modeling of the lateral vehicle dynamics presented in:

    @article{DaLio2020Mental,
      author={Da Lio, Mauro and Donà, Riccardo and Papini, Gastone Pietro Rosati and Biral, Francesco and Svensson, Henrik},
      journal={IEEE Access}, 
      title={A Mental Simulation Approach for Learning Neural-Network Predictive Control (in Self-Driving Cars)}, 
      year={2020},
      volume={8},
      number={},
      pages={192041-192064},
      doi={10.1109/ACCESS.2020.3032780}
    }

### Control Steer Car Parking
The file presents a neural network for the control of the steering angle for parking maneuvers presented in:

    @article{Pagot2023Fast,
      author={Pagot, Edoardo and Piccinini, Mattia and Bertolazzi, Enrico and Biral, Francesco},
      journal={IEEE Access}, 
      title={Fast Planning and Tracking of Complex Autonomous Parking Maneuvers With Optimal Control and Pseudo-Neural Networks}, 
      year={2023},
      volume={11},
      number={},
      pages={124163-124180},
      doi={10.1109/ACCESS.2023.3330431}
    }

### Control Steer Artificial Race Driver
The file presents a neural network for the control of the steering angle for an artificial race driver
presented in:

    @article{piccinini2023physics,
      author={Piccinini, Mattia and Taddei, Sebastiano and Larcher, Matteo and Piazza, Mattia and Biral, Francesco},
      journal={IEEE Access}, 
      title={A Physics-Driven Artificial Agent for Online Time-Optimal Vehicle Motion Planning and Control}, 
      year={2023},
      volume={11},
      number={},
      pages={46344-46372},
      keywords={Motion planning;Biological system modeling;Vehicle dynamics;Tracking;Load modeling;Artificial neural networks;Computational modeling;Neural networks;Autonomous racing;model learning;model predictive control (MPC);motion planning;neural networks;trajectory optimization},
      doi={10.1109/ACCESS.2023.3274836}
    }

### Vehicle Mass Estimation
The file presents a neural network for the estimation of the vehicle mass.

### Road Friction Aware ABS
The file presents a neural network for the estimation of the road friction coefficient.

## Other Applications

### Equation Learner Network
The file presents a simple example of the Equation Learner Network via nnodely. The core ideas is presented in:

    @article{perezvilleda2023learning,
        title = {Learning and extrapolation of robotic skills using task-parameterized equation learner networks},
        journal = {Robotics and Autonomous Systems},
        volume = {160},
        pages = {104309},
        year = {2023},
        issn = {0921-8890},
        doi = {https://doi.org/10.1016/j.robot.2022.104309},
        author = {Hector Perez-Villeda and Justus Piater and Matteo Saveriano},
        keywords = {Learning from demonstration, Learning parameterized skills, Skill generalization and extrapolation, Equation learner neural networks}
    }

### Sobolev Learning
The file presents a simple implementation and test of a Sobolev learning via nnodely. The core ideas is presented in:

    @misc{czarnecki2017sobolev,
          title={Sobolev Training for Neural Networks}, 
          author={Wojciech Marian Czarnecki and Simon Osindero and Max Jaderberg and Grzegorz Świrszcz and Razvan Pascanu},
          year={2017},
          eprint={1706.04859},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/1706.04859}, 
    }

### Physics-Informed Neural Networks
The file presents a simple implementation for solving the Burger's equation using Physics-Informed Neural Networks via nnodely. The main idea of PINN is presented in:

    @article{RAISSI2019686,
        title = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
        journal = {Journal of Computational Physics},
        volume = {378},
        pages = {686-707},
        year = {2019},
        issn = {0021-9991},
        doi = {https://doi.org/10.1016/j.jcp.2018.10.045},
        author = {M. Raissi and P. Perdikaris and G.E. Karniadakis},
    }
