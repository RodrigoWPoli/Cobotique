"""cobot.gripper against a fake Robotiq URCap server (same text protocol as port 63352)."""
import socket
import threading
import time

import pytest

from cobot.gripper import AsyncGripper, RobotiqGripper


class FakeURCap:
    """Answers SET/GET like the Robotiq URCap. Activation (ACT 0 -> 1) takes a few status polls."""

    def __init__(self, active):
        self.vars = {'ACT': int(active), 'STA': 3 if active else 0, 'POS': 0, 'SPE': 0, 'FOR': 0,
                     'GTO': 0, 'OBJ': 3, 'ATR': 0}
        self.log = []
        self._polls = 0
        self.srv = socket.create_server(('127.0.0.1', 0))
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.srv.accept()
        with conn:
            for line in conn.makefile('r'):
                words = line.split()
                self.log.append(line.strip())
                if words[0] == 'SET':
                    for k, v in zip(words[1::2], words[2::2]):
                        self.vars[k] = int(v)
                        if k == 'ACT':
                            self.vars['STA'] = 0
                            self._polls = 0
                    conn.sendall(b'ack\n')
                else:
                    if words[1] == 'STA' and self.vars['ACT'] == 1 and self.vars['STA'] != 3:
                        self._polls += 1
                        self.vars['STA'] = 3 if self._polls >= 3 else 1
                    conn.sendall(f'{words[1]} {self.vars[words[1]]}\n'.encode())


def test_already_active_gripper_is_not_reactivated():
    urcap = FakeURCap(active=True)
    g = RobotiqGripper('127.0.0.1', port=urcap.port)
    assert not any('ACT' in line for line in urcap.log)     # activating would make it move
    g.set_closed(True)
    assert urcap.vars['POS'] == 255 and urcap.vars['GTO'] == 1
    assert urcap.vars['SPE'] == g.speed and urcap.vars['FOR'] == g.force
    g.set_closed(False)
    assert urcap.vars['POS'] == 0
    g.close()


def test_inactive_gripper_is_reset_then_activated():
    urcap = FakeURCap(active=False)
    g = RobotiqGripper('127.0.0.1', port=urcap.port)
    sets = [line for line in urcap.log if line.startswith('SET')]
    assert sets[0] == 'SET ACT 0 ATR 0' and sets[1] == 'SET ACT 1'
    assert urcap.vars['STA'] == 3
    g.close()


def test_missing_urcap_gives_a_clear_error():
    s = socket.create_server(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()                                                # nothing listens there now
    with pytest.raises(RuntimeError, match='Robotiq URCap'):
        RobotiqGripper('127.0.0.1', port=port, timeout=0.5)


class SlowGripper:
    def __init__(self):
        self.sent = []

    def set_closed(self, closed):
        time.sleep(0.1)
        self.sent.append(closed)

    def close(self):
        pass


def test_async_gripper_never_blocks_and_only_sends_what_is_still_needed():
    g = SlowGripper()
    claw = AsyncGripper(g)
    t = time.perf_counter()
    claw.request(True)                     # starts sending "close" (busy 0.1 s)
    time.sleep(0.02)
    for closed in (False, True, False):    # changes of mind while busy: only the last counts
        claw.request(closed)
    claw.request(False)                    # repeated request: nothing new to send
    assert time.perf_counter() - t < 0.05  # the caller never waits for the gripper
    time.sleep(0.4)
    assert g.sent == [True, False] and claw.state is False
    claw.close()
