import random

from llmresfi.scheduler import StageTracker


def test_inference_transitions():
    t = StageTracker("inference")
    assert t.stage == "prefill"
    t.on_prefill_done()
    assert t.stage == "first_token"
    t.on_first_token_emitted()
    assert t.stage == "decode"
    assert t.token_step == 0
    t.inc_decode()
    assert t.token_step == 1


def test_training_transitions():
    t = StageTracker("training")
    assert t.stage == "forward"
    t.on_forward_done()
    assert t.stage == "backward"
    t.on_backward_done()
    assert t.stage == "optimizer_step"
    t.on_optimizer_done()
    assert t.stage == "comm"
    t.on_comm_done()
    assert t.stage == "forward"
    assert t.step == 1


def test_matches_decode_random():
    t = StageTracker("inference")
    t.on_prefill_done()
    t.on_first_token_emitted()
    t.set_random_decode_target(10, random.Random(0))
    assert t.matches("decode", "random") == (t.token_step == t._rand_decode_target)
