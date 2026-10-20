import datetime as _dt
import time


def parse_stop_at(s):
    if s is None:
        return None
    return _dt.datetime.fromisoformat(s).timestamp()


class Deadline:
    def __init__(self, stop_at, reserve_min=20.0):
        self.t = parse_stop_at(stop_at)
        self.reserve = 60.0 * float(reserve_min)
        self.longest_epoch = 0.0

    @property
    def active(self):
        return self.t is not None

    def remaining(self):
        return None if self.t is None else self.t - time.time()

    def epoch_done(self, seconds):
        self.longest_epoch = max(self.longest_epoch, float(seconds))

    def room_for_epoch(self, final_eval_s=0.0):
        if self.t is None:
            return True
        need = self.longest_epoch + max(self.reserve, float(final_eval_s))
        return time.time() + need <= self.t


def add_deadline_args(p):
    p.add_argument("--stop_at", type=str, default=None,
                   help="ISO date-time (e.g. 2026-10-01T23:45): stop training in time to finish "
                        "the final evaluation before it")
    p.add_argument("--eval_reserve_min", type=float, default=20.0,
                   help="minutes kept for the final evaluation when --stop_at is set")
