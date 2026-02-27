# DB_games_linkedin

This repository contains only final databases (no scripts):

- `data/zip.json`: puzzles without solution (named `Zip #XXX - YYYY-MM-DD`)
- `data/zip_solution.json`: same puzzles with a single final `solution` field
- `data/queen.json`: puzzles without solution (named `Queens #XXX - YYYY-MM-DD`)
- `data/queen_solution.json`: same puzzles with a single final `solution` field

All files use the same root schema:

```json
{
  "game": "zip|queens",
  "version": 1,
  "puzzles": []
}
```
