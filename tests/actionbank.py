"""Holds testing for the action bank.

This lays out simple test cases which can be used to test the action bank.
"""
from mentalgym import ActionBank

################
#  Mock Data   #
################
# The first set of actions are *inputs* in the dataset.
data_input_one = {
    'id': 'column_0',
    'source': True,
    'input': None
}
data_input_two = {
    'id': 'column_1',
    'source': True,
    'input': None
}
action_one = {
    'id': 'steve',
    'source': False,
    'input': ['column_0']
}
action_two = {
    'id': 'bob',
    'source': False,
    'input': ['column_0', 'column_1']
}
# Action manifest
action_manifest = [
    data_input_one,
    data_input_two,
    action_one,
    action_two
]

err_msg_header = "Action Bank "

def test_ab():
    """Test the action bank."""
    err_msg = f"""Action Bank Init Error:
    _action_bank_directory property not set correctly.
    Expected value: {d}
    Actual value: {ab._action_bank_directory}
    """

    with tempfile.TemporaryDirectory() as d:
        action_bank = ActionBank(
            action_bank_directory = d
        )
        # 1. Check to ensure the init functioned correctly
        assert ab._action_bank_directory == d, err_msg
        # 2. Check to ensure that the default set of actions
        #   are created
        default = None
        err_msg = """Action Bank Init Error:
        The default set of actions was created incorrectly.
        Expected value: {}
        Actual value: {}
        """
        assert ab._action_manifest == default
