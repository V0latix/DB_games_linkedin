# DB_games_linkedin

This repository contains only two Zip databases:

- `data/zip.json`: puzzles without solution (named `Zip #XXX - YYYY-MM-DD`)
- `data/zip_solution.json`: same puzzles with a single final `solution` field

Both files use the same root schema:

```json
{
  "game": "zip",
  "version": 1,
  "puzzles": []
}
```
