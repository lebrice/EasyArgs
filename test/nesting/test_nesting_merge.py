from simple_parsing import ConflictResolution


def test_hparam_use_case(silent, HyperParameters, TaskHyperParameters):
    hparams = HyperParameters.setup(
        "--num_layers 5 6 7", conflict_resolution_mode=ConflictResolution.ALWAYS_MERGE
    )
    assert isinstance(hparams, HyperParameters)
    # print(hparams.get_help_text())
    assert hparams.gender.num_layers == 5
    assert hparams.age_group.num_layers == 6
    assert hparams.personality.num_layers == 7

    assert hparams.gender.num_units == 32
    assert hparams.age_group.num_units == 64
    assert hparams.personality.num_units == 8

    assert hparams.gender.use_likes is True
    assert hparams.age_group.use_likes is True
    assert hparams.personality.use_likes is False
