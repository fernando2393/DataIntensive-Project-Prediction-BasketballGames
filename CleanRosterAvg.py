import glob
import os


def clean_season_averages(season):
    try:
        os.stat("season_averages/" + str(season) + "/rosters_old")
    except:
        os.mkdir("season_averages/" + str(season) + "/rosters_old")
    files = glob.glob('season_averages/' + str(season) + '/*.json', recursive=False)
    for file in files:
        f = file.split("/")[-1]
        os.replace(file, "season_averages/" + str(season) + "/rosters_old/" + str(f))


def main():
    for i in range(1990, 2019):
        clean_season_averages(i)


if __name__ == "__main__":
    main()
