from mentalgym.utils.reward import build_reward_function

experiment_space = None
function_space = None

# In this experiment space the closest distance is 'd'
# Here there are three sets of input, with d of 0, 0.5, 1
d = (0, 0.5, 1)
# The three levels of monotonic reward should return these values.
monotonic_reward = [1, .606531, .36788]
# The two levels of connection reward should return
connection_reward = [0, 10]
# The completion reward should be a linear combination
# Accuracies: 50, 70, 90
# Values: .5*100 + 20, .7*100 + 20, .9*100 + 20

reward_function_sets = [
    ([],)
    (['monotonic']),
    (['connection']),
    (['completion']),
    (['monotonic','connection']),
    (['connection','completion'])
]
@pytest.mark.parametrize('reward_functions,output',reward_function_sets)
def test_reward_function(reward_functions,output):
    pass