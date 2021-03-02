# Mental Gymnastics

This is a repository which houses work put towards an OpenAI compatible reinforcement learning environment to deliver the final project for Deep Learning, Spring 2021.

This is an environment where you can specify a number of atomic *actions* (with code) that you wish an agent to consider.

The agent, beginning with these blocks, will begin experimenting and will add higher level functions into it's action space as it *learns to learn* and evolves to solve your problem.

## The Environment, or Experiment Space

The environment presents a state space (hereafter referred to as Experiment Space) to the agent consisting of a two-dimensional picture image representing an 'Experiment Space' into which the agent will add blocks.

Note that we should provide a base method for reading raw results and should allow exposing that to extend the gym environment. (Negotiable)

The Experiment Space represents inputs and actions as:
* Input nodes as green dots embedded uniformly distant from one another along the first axis of the Experiment Space.
* Output nodes as red dots embedded equidistant from all inputs along the final axis of the Experiment Space.
* All actions are represented by an Emoji (negotiable) representation which uniquely identify the action.
* Lines representing connections made.

A portion of the Experiment Space is reserved for an image representation of the Action Space, which represents available actions in the experiment space by:
* An Emoji (negotiable) representation of the node.
* An arbitrary length list of *pertinent information* for that agent which includes:
    * Recent performance across a number of episodes, depicted as a block of color.
    * Arbitrary statistics as a function of Experiment Space, provided by the user, and depicted as blocks of color.

Although the default Experiment Space is a two dimensional Euclidean surface there is no reason to limit to a two-dimensional space.
Adding more dimensions simply means that using something like t-SNE to reduce to a two-dimensional manifold might be required.
More discussion required here.

**Here is a placeholder of the image**

## The Action Space

The action the agent performs is of the form <i,j,r,m>:

* The i represents the action the agent wishes to draw from the action table.
* The j represents the input node to branch from, and use as input.
* The r represents the *sink radius* (either an integer, meaning equally in all directions, or a vector of floats meaning 'this far along every dimension'.)
* The m represents the n length vector representing the *physical location in Experiment Space*.

When the agent adds a node into the environment it is initially rewarded simply for placing links from source nodes (representing the input features) to sink nodes (representing your loss function, which defaults to CE for classification, also negotiable). Agents which place nodes which do not connect with anything else do receive some very minor reward which monotonically decreases with distance to the sink.

The reward system is linearly superimposed with an optional reward function that *you write*, which can access Experiment Space and Action space when calculating.
A default reward system is embedded which rewards the agent with logarithmically increasing reward (negotiable) based on reduced variance in predictions.

Additionally, the agent is *only* rewarded if it places an action which does not already exist in the space in the same location; if no changes are made to the DAG, no rewards are earned.

Within the environment the bank of available actions draws the top *m* (based on score metric) and random function *m* actions from previously defined actions (uniform random by default).
These previously defined actions are *created* by the agent.

*A conversation about how to create and store the layers is in order.

## Running Episodes, or an Experiment

In a single Experiment, an agent may run many trials. In a trial, or an episode, an agent will place *n* nodes before terminating the trial.
Limiting this to small numbers is likely to produce more generalizable results.

When an agent places a node it changes the in place net by adding a new node at that location and aggregating all the 'in radius nodes' in the manner specified by the action. If inputs can be simplified easily, they are (negotiable, this would add a layer of complexity, but would likely reduce computational burden.)

At the end of an episode the *net structure* that the agent created is saved as a single new action into the Action Space.


## Curating the Action Space

By default this stores all actions, though actions which have *gone stale* (i.e. haven't been in the leaderboard for some time and no layer in the leader board relies on them) will be pruned, and all their descendants will be removed as well.

This will help to limit the breadth of available actions.
