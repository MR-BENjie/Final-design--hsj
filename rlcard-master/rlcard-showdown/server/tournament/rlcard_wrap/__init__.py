import rlcard
from .leduc_holdem_random_model import LeducHoldemRandomModelSpec
from .doudizhu_random_model import DoudizhuRandomModelSpec
from .red_10_random_model import Red_10RandomModelSpec,Red_10TrainedModelSpec

# Register Leduc Holdem Random Model
rlcard.models.registration.model_registry.model_specs['leduc-holdem-random'] = LeducHoldemRandomModelSpec()

# Register Doudizhu Random Model
rlcard.models.registration.model_registry.model_specs['doudizhu-random'] = DoudizhuRandomModelSpec()

rlcard.models.registration.model_registry.model_specs['red_10-random'] = Red_10RandomModelSpec()

rlcard.models.registration.model_registry.model_specs['red_10-trained'] = Red_10TrainedModelSpec()

# The models we are concerned
MODEL_IDS = {}
MODEL_IDS['red_10'] = [
        'red_10-rule-v2',
        'red_10-rule-v1',
        ]


