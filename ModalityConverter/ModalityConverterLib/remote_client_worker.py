"""Authenticated client-side bridge; input/output are local NumPy .npy files."""
import argparse
import os
import sys
import time
import zipfile


def report(percent, message):
    print("MODALITY_CONVERTER_PROGRESS:{}:{}".format(percent, message), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("server", "input", "output", "model", "module", "device"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mask")
    parser.add_argument("--certificate", default="")
    parser.add_argument("--show-previews", default="1")
    parser.add_argument("--input-preprocessed", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("MODALITY_CONVERTER_BEARER_TOKEN", "")
    if not token:
        raise RuntimeError("Remote bearer token is missing")
    import requests
    from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
    headers = {"Authorization": "Bearer " + token}
    verify = args.certificate or True
    url = args.server.rstrip("/") + "/api/v1/inference"
    openFiles = []
    try:
        fields = {
            "model": args.model,
            "module": args.module,
            "device": args.device,
            "show_previews": "true" if args.show_previews == "1" else "false",
            "input_preprocessed": "true" if args.input_preprocessed else "false",
        }
        inputFile = open(args.input, "rb")
        openFiles.append(inputFile)
        fields["input_file"] = ("input.npy", inputFile, "application/octet-stream")
        if args.mask:
            maskFile = open(args.mask, "rb")
            openFiles.append(maskFile)
            fields["mask_file"] = ("mask.npy", maskFile, "application/octet-stream")

        encoder = MultipartEncoder(fields=fields)
        lastUploadPercent = [-1]
        lastUploadReport = [0.0]
        def reportUpload(monitor):
            # MultipartEncoderMonitor invokes this for every small HTTP read.
            # Throttle the callback's work and progress messages for large volumes.
            now = time.monotonic()
            percent = min(10, int(10 * monitor.bytes_read / max(1, monitor.len)))
            if percent != lastUploadPercent[0] or now - lastUploadReport[0] >= 1.0:
                lastUploadPercent[0] = percent
                lastUploadReport[0] = now
                report(percent, "Uploading volume to remote site ({}%)".format(percent * 10))

        monitor = MultipartEncoderMonitor(encoder, reportUpload)
        headers["Content-Type"] = monitor.content_type
        report(0, "Uploading volume to remote site")
        try:
            response = requests.post(url, headers=headers, data=monitor, timeout=None, verify=verify)
        except requests.RequestException as exc:
            raise RuntimeError("Volume upload failed before the remote site accepted the job: {}".format(exc)) from exc
    finally:
        for fileObject in openFiles:
            fileObject.close()
    response.raise_for_status()
    job_id = response.json()["job_id"]
    status_url = args.server.rstrip("/") + "/api/v1/inference/" + job_id
    consecutiveStatusFailures = 0
    while True:
        try:
            status = requests.get(status_url, headers=headers, timeout=(10, 60), verify=verify)
            status.raise_for_status()
        except requests.RequestException as exc:
            consecutiveStatusFailures += 1
            if consecutiveStatusFailures >= 10:
                raise RuntimeError("Cannot read remote inference status after 10 retries: {}".format(exc))
            report(0, "Connection interrupted; retry {}/10: {}".format(consecutiveStatusFailures, exc))
            time.sleep(min(2 + consecutiveStatusFailures, 8))
            continue
        consecutiveStatusFailures = 0
        data = status.json()
        overallProgress = min(98, 10 + int(data["progress"] * 0.88))
        report(overallProgress, data["message"])
        if data["finished"]:
            if data["error"]:
                raise RuntimeError(data["error"])
            break
        time.sleep(1)
    report(99, "Downloading inference result...")
    result = requests.get(status_url + "/result", headers=headers, timeout=(15, None), stream=True, verify=verify)
    result.raise_for_status()
    with open(args.output, "wb") as output:
        for chunk in result.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    if args.show_previews == "1":
        report(99, "Downloading preview volumes...")
        previewResponse = requests.get(status_url + "/previews", headers=headers,
                                       timeout=(15, None), stream=True, verify=verify)
        if previewResponse.status_code == 404:
            previewResponse.close()
            previewResponse = None
        else:
            previewResponse.raise_for_status()
        previewArchivePath = os.path.join(os.path.dirname(args.output), "previews.zip")
        if previewResponse is not None:
            with open(previewArchivePath, "wb") as archiveFile:
                for chunk in previewResponse.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archiveFile.write(chunk)
            previewDir = os.path.join(os.path.dirname(args.output), "previews")
            os.makedirs(previewDir, exist_ok=True)
            with zipfile.ZipFile(previewArchivePath, "r") as archive:
                for member in archive.infolist():
                    filename = os.path.basename(member.filename)
                    if filename != member.filename or not filename.endswith(".npy"):
                        continue
                    with archive.open(member) as source, open(os.path.join(previewDir, filename), "wb") as target:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            target.write(chunk)
            os.remove(previewArchivePath)
    report(100, "Remote inference completed")


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print("MODALITY_CONVERTER_ERROR:" + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
