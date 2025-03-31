import datetime as dt

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():

    # Load crossing data (includes Cassini location and boundary dynamic pressure)
    crossings_data = pd.read_csv("./cassini_crossing_list_with_trajectory_DP.txt")

    # Only include crossings were MP is followed directly by BS
    mask = (crossings_data["Typecross"] == "MP") & (crossings_data["Typecross"].shift(-1) == "BS")
    crossings_data = crossings_data[mask | mask.shift(1)]

    bow_shock_crossings = crossings_data.loc[crossings_data["Typecross"] == "BS"]
    magnetopause_crossings = crossings_data.loc[crossings_data["Typecross"] == "MP"]

    bow_shock_crossings["crossing_time"] = bow_shock_crossings.apply(
        convert_to_datetime, axis=1
    )
    magnetopause_crossings["crossing_time"] = magnetopause_crossings.apply(
        convert_to_datetime, axis=1
    )

    bow_shock_dynamic_pressure = calculate_bow_shock_dynamic_pressure(
        bow_shock_crossings
    ).tolist()
    magnetopause_dynamic_pressure = calculate_magnetopause_dynamic_pressure(
        magnetopause_crossings
    )

    # Load LFE detections
    lfe_data = pd.read_csv("./LFEs_joined.csv", parse_dates=["start", "end"])

    lfe_start_times = np.array(lfe_data["start"].tolist())
    lfe_end_times = np.array(lfe_data["end"].tolist())

    # For each MP crossing (keep flexible as could be BS or some offset), find
    # the nearest LFE before, and the nearest LFE after.
    # Make a bar plot with time on the X axis, and some other parameter on the
    # Y axis (MP index, DP, etc.)
    comparison_lfes = []
    comparison_metric = magnetopause_crossings["crossing_time"]

    for comparison_time in comparison_metric:

        # Check if crossing_time is inside any LFE (start ≤ crossing_time ≤ end)
        inside_lfe = (
            (lfe_start_times <= comparison_time) & (lfe_end_times >= comparison_time)
        ).any()

        if not inside_lfe:
            lfe_index_before = np.searchsorted(lfe_end_times, comparison_time) - 1
            lfe_index_after = np.searchsorted(lfe_start_times, comparison_time)

            comparison_lfes.append((lfe_index_before, lfe_index_after))

        else:
            # If we are crossing during an LFE we just want to plot that one
            inside_lfe_index = np.where(
                (lfe_start_times <= comparison_time)
                & (lfe_end_times >= comparison_time)
            )[0]

            if (len(inside_lfe_index)) != 1:
                raise ValueError()

            comparison_lfes.append((inside_lfe_index))

    fig, ax = plt.subplots()

    y_values = list(range(len(magnetopause_crossings)))
    sort_by = "MP DP" # "BS DP", "MP DP", "BS DP Calc", "MP DP Calc"

    match sort_by:
        case "BS DP":
            # Bow shock DP from file
            y_values = np.argsort(bow_shock_crossings["inf_DP"])

        case "MP DP":
            # Magnetopause DP from file
            y_values = np.argsort(magnetopause_crossings["inf_DP"])

        case "BS DP Calc":
            # Bow shock DP from function
            y_values = np.argsort(bow_shock_dynamic_pressure)

        case "MP DP Calc":
            # Magnetopause DP from function
            pass

        case _:
            pass



    # Formatting
    ax.axvline(0, ymin=np.min(y_values), ymax=np.max(y_values), color="black", ls="dashed")
    ax.set_ylim(np.min(y_values), np.max(y_values))
    ax.set_xlim(-600, 600)

    # Adding LFEs
    patches = []
    patch_height = 1

    for y, lfe_index_group, comparison_time in zip(y_values, comparison_lfes, comparison_metric):

        # if only 1, crossing is during an LFE
        if len(lfe_index_group) == 1:
            i = lfe_index_group[0]
            lfe_start = (lfe_start_times[i] - comparison_time).total_seconds() / 3600
            lfe_end = (lfe_end_times[i] - comparison_time).total_seconds() / 3600

            rect = matplotlib.patches.Rectangle(
                (lfe_start, y), lfe_end - lfe_start, patch_height, color="black"
            )
            patches.append(rect)

        else:
            i = lfe_index_group[0]

            lfe_start = (lfe_start_times[i] - comparison_time).total_seconds() / 3600
            lfe_end = (lfe_end_times[i] - comparison_time).total_seconds() / 3600

            rect = matplotlib.patches.Rectangle(
                (lfe_start, y), lfe_end - lfe_start, patch_height, color="blue"
            )
            patches.append(rect)

            i = lfe_index_group[1]

            lfe_start = (lfe_start_times[i] - comparison_time).total_seconds() / 3600
            lfe_end = (lfe_end_times[i] - comparison_time).total_seconds() / 3600

            rect = matplotlib.patches.Rectangle(
                (lfe_start, y), lfe_end - lfe_start, patch_height, color="blue"
            )
            patches.append(rect)

    [ax.add_patch(p) for p in patches]

    ax.set_xlabel("Time Difference from Magnetopause Crossing [hours]")

    plt.show()


def convert_to_datetime(row):
    return dt.datetime.strptime(
        f"{row['Yearcross']} {row['DOYcross']} {row['Hourcross']} {row['Minutecross']}",
        "%Y %j %H %M",
    )


def calculate_bow_shock_dynamic_pressure(crossings_data):
    epsilon = 0.84
    c1 = 15
    c2 = 5.4

    saturn_radius = 60_269  # km

    x_crossings = crossings_data["X_KSM(km)"]
    y_crossings = crossings_data["Y_KSM(km)"]
    z_crossings = crossings_data["Z_KSM(km)"]

    crossing_positions = np.array([x_crossings, y_crossings, z_crossings])

    r_crossings = np.sqrt(np.sum(crossing_positions**2))
    theta_crossings = np.arccos(x_crossings / r_crossings)

    tmp_a = (r_crossings / saturn_radius) * (1 + (epsilon * np.cos(theta_crossings)))
    tmp_b = (1 + epsilon) * c1

    return (tmp_a / tmp_b) ** (-c2)


def calculate_magnetopause_dynamic_pressure(crossings_data):
    return [np.nan] * len(crossings_data)


if __name__ == "__main__":
    main()
