import pandas as pd

########################################
# 1. Load Data                         # 
########################################
sub_df = pd.read_csv("raw_data/SampleSubmissionStage2.csv")
seed_df = pd.read_csv("raw_data/MNCAATourneySeeds.csv")
reg_season_df = pd.read_csv("raw_data/MRegularSeasonDetailedResults.csv")
ordinals_df = pd.read_csv("raw_data/MMasseyOrdinals.csv")


########################################
# 2. Clean, Preprocess & FILTER 2026   # 
########################################
sub_df["Season"] = sub_df["ID"].apply(lambda x: int(x.split("_")[0]))
sub_df["TeamA"] = sub_df["ID"].apply(lambda x: int(x.split("_")[1]))
sub_df["TeamB"] = sub_df["ID"].apply(lambda x: int(x.split("_")[2]))

sub_df = sub_df[sub_df["TeamA"] < 2000]
sub_df = sub_df[sub_df["TeamB"] < 2000]

# Filter for this season
sub_df = sub_df[sub_df["Season"] == 2026]
reg_season_df = reg_season_df[reg_season_df["Season"] == 2026]
ordinals_df = ordinals_df[ordinals_df["Season"] == 2026]
seed_df = seed_df[seed_df["Season"] == 2026] 

# Basic cleaning for Seeds
seed_df["Seed"] = seed_df["Seed"].str.extract(r"(\d+)").astype(int)

# Winning and losing scores
win_scores = reg_season_df[["Season", "WTeamID", "WScore"]].rename(columns={"WTeamID": "TeamID", "WScore": "Points"})
loss_scores = reg_season_df[["Season", "LTeamID", "LScore"]].rename(columns={"LTeamID": "TeamID", "LScore": "Points"})
all_scores = pd.concat([win_scores, loss_scores])

# Group by Season and TeamID to get the average points scored
avg_pts = all_scores.groupby(["Season", "TeamID"])["Points"].mean().reset_index()
avg_pts.rename(columns={"Points": "AvgPtsScored"}, inplace=True)

# Winning and losing offensive rebounds 
win_off_rebounds = reg_season_df[["Season", "WTeamID", "WOR"]].rename(columns={"WTeamID": "TeamID", "WOR": "OR"})
loss_off_rebounds = reg_season_df[["Season", "LTeamID", "LOR"]].rename(columns={"LTeamID": "TeamID", "LOR": "OR"})
all_off_rebounds = pd.concat([win_off_rebounds, loss_off_rebounds])

# Group by Season and TeamID to get the average offensive rebounds 
avg_or = all_off_rebounds.groupby(["Season", "TeamID"])["OR"].mean().reset_index()
avg_or.rename(columns={"OR": "AvgOR"}, inplace=True)

# Winning and losing defensive rebounds 
win_def_rebounds = reg_season_df[["Season", "WTeamID", "WDR"]].rename(columns={"WTeamID": "TeamID", "WDR": "DR"})
loss_def_rebounds = reg_season_df[["Season", "LTeamID", "LDR"]].rename(columns={"LTeamID": "TeamID", "LDR": "DR"})
all_def_rebounds = pd.concat([win_def_rebounds, loss_def_rebounds])

# Group by Season and TeamID to get the average defensive rebounds 
avg_dr = all_def_rebounds.groupby(["Season", "TeamID"])["DR"].mean().reset_index()
avg_dr.rename(columns={"DR": "AvgDR"}, inplace=True)

# Winning and losing free throws attempted 
win_ft_att = reg_season_df[["Season", "WTeamID", "WFTA"]].rename(columns={"WTeamID": "TeamID", "WFTA": "FTA"})
loss_ft_att = reg_season_df[["Season", "LTeamID", "LFTA"]].rename(columns={"LTeamID": "TeamID", "LFTA": "FTA"})
all_ft_att = pd.concat([win_ft_att, loss_ft_att])

# Group by Season and TeamID to get the average free throws attempted 
avg_fta = all_ft_att.groupby(["Season", "TeamID"])["FTA"].mean().reset_index()
avg_fta.rename(columns={"FTA": "AvgFTA"}, inplace=True)

# Winning and losing free throws made 
win_ft_made = reg_season_df[["Season", "WTeamID", "WFTM"]].rename(columns={"WTeamID": "TeamID", "WFTM": "FTM"})
loss_ft_made = reg_season_df[["Season", "LTeamID", "LFTM"]].rename(columns={"LTeamID": "TeamID", "LFTM": "FTM"})
all_ft_made = pd.concat([win_ft_made, loss_ft_made])

# Group by Season and TeamID to get the average free throws made 
avg_ftm = all_ft_made.groupby(["Season", "TeamID"])["FTM"].mean().reset_index()
avg_ftm.rename(columns={"FTM": "AvgFTM"}, inplace=True)

# Massey Ordinals
final_ordinals = ordinals_df.groupby(["Season", "TeamID"])["RankingDayNum"].max().reset_index()
ordinals_df = ordinals_df.merge(final_ordinals, on=["Season", "TeamID", "RankingDayNum"])
avg_ordinal = ordinals_df.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
avg_ordinal.rename(columns={"OrdinalRank": "AvgOrdinal"}, inplace=True)


########################################
# 3. Extract Base Data                 #
########################################
df = sub_df[["Season", "TeamA", "TeamB"]].copy()
df["TeamA"] = df["TeamA"].astype(int)

def merge_team_stats(base_df, stat_df, original_col_name, new_col_name):
    # Merge for Team A
    base_df = base_df.merge(
        stat_df[["Season", "TeamID", original_col_name]], 
        left_on=["Season", "TeamA"], 
        right_on=["Season", "TeamID"], 
        how="left"
    ).rename(columns={original_col_name: f"TeamA_{new_col_name}"}).drop(columns=["TeamID"])
    
    # Merge for Team B
    base_df = base_df.merge(
        stat_df[["Season", "TeamID", original_col_name]], 
        left_on=["Season", "TeamB"], 
        right_on=["Season", "TeamID"], 
        how="left"
    ).rename(columns={original_col_name: f"TeamB_{new_col_name}"}).drop(columns=["TeamID"])
    
    return base_df


df = merge_team_stats(df, seed_df, "Seed", "Seed")
df["TeamA_Seed"] = df["TeamA_Seed"].fillna(17)
df["TeamB_Seed"] = df["TeamB_Seed"].fillna(17)

df = merge_team_stats(df, avg_pts, "AvgPtsScored", "AvgPts")
df = merge_team_stats(df, avg_or, "AvgOR", "AvgOR")
df = merge_team_stats(df, avg_dr, "AvgDR", "AvgDR")
df = merge_team_stats(df, avg_fta, "AvgFTA", "AvgFTA")
df = merge_team_stats(df, avg_ftm, "AvgFTM", "AvgFTM")
df = merge_team_stats(df, avg_ordinal, "AvgOrdinal", "Ordinal")


########################################
# 4. Calculate Final Modeling Features #
########################################

# Initialize the empty dataframe before assigning columns to it
feature_df = pd.DataFrame()

feature_df["Season"] = df["Season"]
feature_df["SeedDiff"] = df["TeamA_Seed"] - df["TeamB_Seed"]
feature_df["AvgPtsDiff"] = df["TeamA_AvgPts"] - df["TeamB_AvgPts"]
feature_df["AvgORDiff"] = df["TeamA_AvgOR"] - df["TeamB_AvgOR"] 
feature_df["AvgDRDiff"] = df["TeamA_AvgDR"] - df["TeamB_AvgDR"] 
feature_df["AvgFTADiff"] = df["TeamA_AvgFTA"] - df["TeamB_AvgFTA"] 
feature_df["AvgFTMDiff"] = df["TeamA_AvgFTM"] - df["TeamB_AvgFTM"]
feature_df["OrdinalDiff"] = df["TeamA_Ordinal"] - df["TeamB_Ordinal"]

#feature_df = feature_df.fillna(0) 

print(feature_df.head())
feature_df.to_csv("clean_data/2026_submission_features.csv", index=False)

