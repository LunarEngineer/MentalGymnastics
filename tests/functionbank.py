"""Holds testing for the function bank.

This lays out simple test cases which can be used to test the function bank.
"""
from mentalgym import FunctionBank
from mentalgym.utils.data import atomic_functions

################
#  Mock Data   #
################
# The first set of actions are *inputs* in the dataset.
# These are represented as input nodes in the experiment
#   space / canvas. These are non-placable actions that
#   the environment starts with. This should be replaced
#   with a 'read_dataset' function that can make these
#   dicts for input datasets.
data_input_one = {
    'id': 'column_0',
    'type': 'source',
    'input': None
}
data_input_two = {
    'id': 'column_1',
    'type': 'source',
    'input': None
}
# The second set of actions are *composed* actions
# These are created by agents during episodes.
action_one = {
    'id': 'steve',
    'type': 'composed',
    'input': ['column_0']
}
action_two = {
    'id': 'bob',
    'type': 'composed',
    'input': ['column_0', 'column_1']
}
# Action manifest
action_manifest = atomic_functions + [
    data_input_one,
    data_input_two,
    action_one,
    action_two
]

err_msg_header = "Action Bank "

def test_action_bank():
    """Test the action bank."""
    err_msg = f"""Action Bank Init Error:
    _action_bank_directory property not set correctly.
    Expected value: {d}
    Actual value: {ab._action_bank_directory}
    """

    with tempfile.TemporaryDirectory() as d:
        # Spin up a new action bank using the temp directory.
        action_bank = ActionBank(
            action_bank_directory = d
        )
        # 1. Check to ensure the init functioned correctly
        assert ab._action_bank_directory == d, err_msg
        # 2. Check to ensure that the default set of actions
        #   are created
        default = action_bank._query('type=="atomic"')
        err_msg = """Action Bank Init Error:
        The default set of actions was created incorrectly.
        Expected value: {}
        Actual value: {}
        """
        assert ab._action_manifest == default
