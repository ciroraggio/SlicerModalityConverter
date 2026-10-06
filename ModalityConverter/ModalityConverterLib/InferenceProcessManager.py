"""Asynchronous QProcess wrapper for the ModalityConverter inference worker."""

import os
import shutil
import uuid

import qt


class InferenceProcessManager(qt.QObject):
    """Owns one worker process and forwards its output as Qt signals."""

    started = qt.Signal(str)
    progressChanged = qt.Signal(int, str)
    stdoutReceived = qt.Signal(str)
    stderrReceived = qt.Signal(str)
    finished = qt.Signal(str, int, str)
    failed = qt.Signal(str, str)
    cancelled = qt.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.workDir = None
        self.runToken = None
        self._stdoutBuffer = ""
        self._stdoutLog = []
        self._stderrLog = []
        self._cancelRequested = False
        self._killTimer = None

    @property
    def isRunning(self):
        return self.process is not None and self.process.state() != qt.QProcess.NotRunning

    def start(self, pythonExecutable, workerPath, inputPath, outputPath, modelKey,
              moduleName, device, extensionRoot, modelCacheDir, maskPath=None,
              showAllFiles=True, pythonPath=None, extraArguments=None):
        if self.isRunning:
            raise RuntimeError("An inference process is already running")
        if not os.path.isfile(workerPath):
            raise FileNotFoundError("Inference worker not found: " + workerPath)
        self.workDir = os.path.dirname(inputPath)
        self.runToken = str(uuid.uuid4())
        self._stdoutBuffer = ""
        self._stdoutLog = []
        self._stderrLog = []
        self._cancelRequested = False

        process = qt.QProcess(self)
        self.process = process
        process.setWorkingDirectory(self.workDir)
        environment = qt.QProcessEnvironment.systemEnvironment()
        if pythonPath:
            previous = environment.value("PYTHONPATH")
            environment.insert("PYTHONPATH", pythonPath + (os.pathsep + previous if previous else ""))
        environment.insert("MODALITY_CONVERTER_MODEL_DIR", modelCacheDir)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._readStdout)
        process.readyReadStandardError.connect(self._readStderr)
        process.finished.connect(self._processFinished)
        process.errorOccurred.connect(self._processError)

        arguments = [workerPath, "--model", modelKey, "--module", moduleName,
                     "--input", inputPath, "--output", outputPath, "--device", device,
                     "--show-previews", "1" if showAllFiles else "0",
                     "--extension-root", extensionRoot]
        if maskPath:
            arguments += ["--mask", maskPath]
        if extraArguments:
            arguments.extend(extraArguments)
        self.started.emit(self.runToken)
        process.start(pythonExecutable, arguments)
        return self.runToken

    def startRemote(self, pythonExecutable, workerPath, context, serverUrl, bearerToken,
                    inputPreprocessed=False):
        """Run the HTTP bridge asynchronously, keeping network I/O off Slicer's UI thread."""
        if self.isRunning:
            raise RuntimeError("An inference process is already running")
        self.workDir = context["workDir"]
        self.runToken = str(uuid.uuid4())
        self._stdoutBuffer, self._stdoutLog, self._stderrLog = "", [], []
        self._cancelRequested = False
        process = qt.QProcess(self)
        self.process = process
        process.setWorkingDirectory(self.workDir)
        environment = qt.QProcessEnvironment.systemEnvironment()
        environment.insert("MODALITY_CONVERTER_BEARER_TOKEN", bearerToken)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._readStdout)
        process.readyReadStandardError.connect(self._readStderr)
        process.finished.connect(self._processFinished)
        process.errorOccurred.connect(self._processError)
        arguments = [workerPath, "--server", serverUrl, "--input", context["inputPath"],
                     "--output", context["outputPath"], "--model", context["modelKey"],
                     "--module", context["moduleName"], "--device", context["device"],
                     "--show-previews", "1" if context["showAllFiles"] else "0"]
        if inputPreprocessed:
            arguments += ["--input-preprocessed"]
        if context["maskPath"]:
            arguments += ["--mask", context["maskPath"]]
        self.started.emit(self.runToken)
        process.start(pythonExecutable, arguments)
        return self.runToken

    @staticmethod
    def _decodeOutput(data):
        try:
            return bytes(data).decode("utf-8", errors="replace")
        except Exception:
            try:
                return data.data().decode("utf-8", errors="replace")
            except Exception:
                return str(data)

    def _readStdout(self):
        if not self.process:
            return
        chunk = self._decodeOutput(self.process.readAllStandardOutput())
        if not chunk:
            return
        self._stdoutLog.append(chunk)
        self.stdoutReceived.emit(chunk)
        self._stdoutBuffer += chunk
        lines = self._stdoutBuffer.splitlines(True)
        self._stdoutBuffer = ""
        if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r")):
            self._stdoutBuffer = lines.pop()
        for line in lines:
            self._parseWorkerLine(line.rstrip())

    def _parseWorkerLine(self, line):
        prefix = "MODALITY_CONVERTER_PROGRESS:"
        if line.startswith(prefix):
            try:
                value, _, message = line[len(prefix):].partition(":")
                self.progressChanged.emit(max(0, min(100, int(value))), message)
                return
            except ValueError:
                pass

    def _readStderr(self):
        if self.process:
            chunk = self._decodeOutput(self.process.readAllStandardError())
            if chunk:
                self._stderrLog.append(chunk)
                self.stderrReceived.emit(chunk)

    def _processError(self, error):
        if self._cancelRequested or not self.process:
            return
        detail = str(self.process.errorString())
        token = self.runToken
        if error == qt.QProcess.FailedToStart:
            process = self.process
            self.process = None
            process.deleteLater()
            self.failed.emit(token, "Could not start inference worker: " + detail)

    def _processFinished(self, exitCode, exitStatus=None):
        process = self.process
        if not process:
            return
        self._readStdout()
        self._readStderr()
        if self._stdoutBuffer:
            self._parseWorkerLine(self._stdoutBuffer.rstrip())
            self._stdoutBuffer = ""
        token = self.runToken
        stdout = "".join(self._stdoutLog)
        stderr = "".join(self._stderrLog)
        wasCancelled = self._cancelRequested
        process.deleteLater()
        self.process = None
        if self._killTimer:
            self._killTimer.stop()
            self._killTimer.deleteLater()
            self._killTimer = None
        self._cancelRequested = False
        if wasCancelled:
            self.cancelled.emit(token)
        elif int(exitCode) == 0 and (exitStatus is None or exitStatus == qt.QProcess.NormalExit):
            self.finished.emit(token, int(exitCode), stdout)
        else:
            detail = stderr.strip() or stdout.strip() or "Worker exited with code {}".format(exitCode)
            self.failed.emit(token, detail)

    def cancel(self):
        if not self.isRunning or self._cancelRequested:
            return
        self._cancelRequested = True
        self.process.terminate()
        self._killTimer = qt.QTimer(self)
        self._killTimer.setSingleShot(True)
        self._killTimer.timeout.connect(self._killIfRunning)
        self._killTimer.start(3000)

    def _killIfRunning(self):
        if self.isRunning:
            self.process.kill()

    def cleanup(self, removeWorkDir=True):
        if self._killTimer:
            self._killTimer.stop()
            self._killTimer.deleteLater()
            self._killTimer = None
        if self.isRunning:
            self._cancelRequested = True
            self.process.kill()
        elif self.process:
            self.process.deleteLater()
            self.process = None
        if removeWorkDir and self.workDir and os.path.isdir(self.workDir):
            shutil.rmtree(self.workDir, ignore_errors=True)
        self.workDir = None

