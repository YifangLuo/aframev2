import logging
import os
import sys
from pathlib import Path
from typing import Optional

from analyze_o2 import utils as analysis_utils
from hermes.typeo import typeo

from bbhnet.io import filter_and_sort_files, fname_re

event_times = [1186302519.8, 1186741861.5, 1187058327.1, 1187529256.5]
event_names = ["GW170809", "GW170814", "GW170818", "GW170823"]


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    num_bins: int = 10000,
    window_length: float = 1.0,
    norm_seconds: Optional[float] = None,
    max_tb: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False
):
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file, mode="w")
        logging.getLogger().addHandler(handler)

    data_dir = Path(data_dir)
    t0s, t0, events, current_event = [], None, [], None
    runs = sorted(map(int, os.listdir(data_dir / "dt-0.0")))
    for run in runs:
        run_dir = data_dir / "dt-0.0" / str(run) / "out"
        for fname in filter_and_sort_files(os.listdir(run_dir)):
            f_t0 = int(fname_re.search(fname).group("t0"))
            f_length = int(fname_re.search(fname).group("length"))

            if t0 is None:
                # this is the first iteration, so initialize everything
                t0 = current_group_t0 = last_t0 = f_t0
                last_length = f_length
                continue
            elif f_t0 > (last_t0 + last_length):
                # a segment has ended, so clock a new segment t0
                logging.debug(f"Detected new segment beginning at GPS time {f_t0}")
                current_group_t0 = f_t0

                if current_event is not None:
                    logging.debug(f"Ending segment with event {current_event}")
                    # if the current segment has an event in it, end
                    # the current analysis period and record the
                    # presence of the event
                    t0s.append(t0)
                    events.append(current_event)
                    current_event = None
                    t0 = f_t0

            last_t0 = f_t0
            last_length = f_length

            # check to see if there's an event contained in this file
            for event_time, event_name in zip(event_times, event_names):
                if f_t0 < event_time < (f_t0 + f_length):
                    logging.debug(
                        "Segment beginning at GPS time {} contains "
                        "event {} at GPS time {}. Ending period beginning "
                        "at GPS time {} to isolate segment.".format(
                            current_group_t0, event_name, event_time, t0
                        )
                    )

                    # there is an event, so end the current analysis
                    # period and start a new on beginning at the
                    # start of this segment
                    events.append(None)
                    current_event = event_name
                    t0s.append(t0)
                    t0 = current_group_t0
                    break

    t0s.append(f_t0 + f_length)
    logging.debug(t0s)

    for event_name, t0, t1 in zip(events, t0s[:-1], t0s[1:]):
        length = t1 - t0
        if event_name is None:
            logging.info(
                "Building max {}s of background samples from timeslides "
                "beginning at GPS time {} and ending at {}".format(
                    max_tb, t0, t1
                )
            )
            fnames, Tb, min_value, max_value = analysis_utils.build_background(
                data_dir,
                write_dir,
                num_bins=num_bins,
                window_length=window_length,
                num_proc=8,
                t0=t0,
                length=length,
                norm_seconds=norm_seconds,
                max_tb=max_tb,
            )
        else:
            logging.info("Insert analysis here")


if __name__ == "__main__":
    main()
