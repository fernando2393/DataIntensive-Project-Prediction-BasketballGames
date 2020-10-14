val df_players = spark.read.json("path")  // Read the players JSON file
val playerIDs = df_players.select("id").cache()  // Get all different player IDs

val df_stats = spark.read.json("path")  // Read the stats JSON file

// Iterate over playerIDs entries per season

// Maybe define as function and do foreach??

val player_games = df_stats.filter($df_stats.player.id === id && $df_stats.game.season === season)
val player_teams = player_games.map(x => (df_stats.team.id, 1))
val player_team = player_teams.reduceByKey(_ + _).maxBy(_._2)  // Reduce by key (team ID) and get the one with more frequency


