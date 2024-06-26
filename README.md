# CASNAV: The Crowd Project

As robots become more prevalent in our daily lives, especially in human-centric environments like airports and tourist places, efficient navigation for robots through dense crowds becomes a critical challenge. One of the biggest challenges we face is the randomized nature of human trajectories. Unlike predictable obstacles, human movements are highly spontaneous and dynamic. This randomness makes it difficult for robots to plan their paths in real time. 

Existing SOTA Approach integrates PPO with human trajectory prediction models to navigate through dense crowds and adjust its path to avoid collisions or intrusions (https://arxiv.org/pdf/2203.01821).

Drawbacks include Once PPO converges to an optimal policy, it stops exploring. Hence, although this approach is efficient and accurate for a specific scenario, it is not flexible and adaptable when there is an environment change from what it was trained on. Additionally, PPO is on-policy, which means it lacks the ability to generalize in different environments. 

Proposed approach is To explore various Deep RL techniques such as Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), and Soft Actor-Critic (SAC) and develop our own strategy considering all the plus points from different policies, compare it with the existing solution using certain benchmarks and come up with a simulation on gazebo for demonstration.


