"""Grippers for a UR5e: a Robotiq 2F-85 / Hand-E through its URCap, or a tool digital output.

    g = RobotiqGripper('100.65.63.93')   # activates it if needed (it then opens and closes once)
    g.set_closed(True)                   # close
    g.close()

Robotiq: the Robotiq URCap runs a text server on the robot, port 63352, independent of the program
ur_rtde runs, so it works next to an RTDEControlInterface. Commands are 'SET VAR value ...' (reply
'ack') and 'GET VAR' (reply 'VAR value'). POS 0 = open, 255 = closed; SPE and FOR are 0-255.

AsyncGripper sends the commands from a thread, so a control loop never waits for the gripper.
"""
import socket
import threading
import time

ROBOTIQ_PORT = 63352
GRIP_SPEED = 128       # 0-255, Robotiq closing/opening speed (about half)
GRIP_FORCE = 100       # 0-255, Robotiq grip force (about 40 %)
STA_ACTIVE = 3         # gripper status: activation done


class RobotiqGripper:
    """Robotiq 2F-85 / Hand-E driven through the Robotiq URCap's socket (port 63352)."""

    def __init__(self, ip, port=ROBOTIQ_PORT, speed=GRIP_SPEED, force=GRIP_FORCE, timeout=2.0,
                 activate_timeout=10.0):
        self.speed, self.force = int(speed), int(force)
        try:
            self._sock = socket.create_connection((ip, port), timeout=timeout)
        except OSError as e:
            raise RuntimeError(f'no Robotiq URCap at {ip}:{port} ({e}). Is the gripper URCap installed '
                               f'and the gripper plugged in?') from e
        self._lock = threading.Lock()
        if self.get('STA') != STA_ACTIVE:
            self._activate(activate_timeout)

    def _ask(self, line):
        with self._lock:
            self._sock.sendall((line + '\n').encode())
            reply = self._sock.recv(1024).decode().strip()
        if not reply:
            raise RuntimeError(f'Robotiq: connection closed after {line!r}')
        return reply

    def set(self, **values):
        """SET several variables in one command, e.g. set(POS=255, SPE=128, FOR=100, GTO=1)."""
        cmd = 'SET ' + ' '.join(f'{k} {int(v)}' for k, v in values.items())
        reply = self._ask(cmd)
        if reply != 'ack':
            raise RuntimeError(f'Robotiq: {cmd!r} answered {reply!r}')

    def get(self, name):
        reply = self._ask(f'GET {name}').split()
        if len(reply) != 2 or reply[0] != name:
            raise RuntimeError(f'Robotiq: GET {name} answered {" ".join(reply)!r}')
        return int(reply[1])

    def _wait(self, name, value, timeout):
        end = time.monotonic() + timeout
        while self.get(name) != value:
            if time.monotonic() > end:
                raise RuntimeError(f'Robotiq: {name} did not reach {value} within {timeout:.0f} s')
            time.sleep(0.05)

    def _activate(self, timeout):
        """Reset, then activate. The gripper opens and closes fully once to find its stroke."""
        print('Robotiq gripper not active: activating (it opens and closes once)...', flush=True)
        self.set(ACT=0, ATR=0)
        self._wait('STA', 0, timeout)
        self.set(ACT=1)
        self._wait('STA', STA_ACTIVE, timeout)

    def set_closed(self, closed):
        """Close (True) or open (False) at GRIP_SPEED / GRIP_FORCE; does not wait for the motion."""
        self.set(POS=255 if closed else 0, SPE=self.speed, FOR=self.force, GTO=1)

    def holding(self):
        """True when the fingers stopped on an object while closing (OBJ = 2)."""
        return self.get('OBJ') == 2

    def close(self):
        self._sock.close()


class ToolDOGripper:
    """Gripper switched by a tool digital output (the simulated gripper of cobot.Robot; works in URSim)."""

    def __init__(self, ip, output=0):
        from rtde_io import RTDEIOInterface
        self.output = output
        self._io = RTDEIOInterface(ip)

    def set_closed(self, closed):
        self._io.setToolDigitalOut(self.output, bool(closed))

    def close(self):
        self._io.disconnect()


class AsyncGripper:
    """Sends open/close requests to a gripper from a thread. Only the latest request counts: asking
    close, open, close quickly sends at most what is still needed. Errors are printed, not raised."""

    def __init__(self, gripper):
        self._g = gripper
        self._want = None          # None: nothing requested yet
        self._sent = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name='gripper', daemon=True)
        self._thread.start()

    def request(self, closed):
        with self._cv:
            self._want = bool(closed)
            self._cv.notify()

    @property
    def state(self):
        """Last state actually sent to the gripper (None before the first command)."""
        return self._sent

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stop or self._want != self._sent)
                if self._stop:
                    return
                want = self._want
            try:
                self._g.set_closed(want)
                self._sent = want
                print(f'Gripper {"closed" if want else "opened"}', flush=True)
            except Exception as e:
                print(f'Gripper command failed: {e}', flush=True)
                time.sleep(0.5)    # don't spin on a broken connection

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2.0)
        self._g.close()
