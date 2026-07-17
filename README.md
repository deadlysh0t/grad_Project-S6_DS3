# Walmart Weekly Sales Predictor

A self-contained Docker image that serves weekly sales predictions using a
trained Random Forest model (R2 = 0.9631 on held-out test data), built from
the Walmart Kaggle "Store Sales Forecasting" dataset.

Docker Hub: https://hub.docker.com/r/ibrahimessam0/walmart-sales-predictor

## Run it

```bash
docker pull ibrahimessam0/walmart-sales-predictor:1.0
docker run -p 8080:8080 ibrahimessam0/walmart-sales-predictor:1.0
```

The API is now available at `http://localhost:8080`.

## Get a prediction

The image serves MLflow's standard model-serving endpoint,
`POST /invocations`, using the `dataframe_split` request format.

**Linux / Mac / Git Bash:**
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["Store","Dept","IsHoliday","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment","Size","Year","Month","Week","Quarter","Day","Type_B","Type_C"],
          "data": [[1,1,false,42.31,2.572,0,0,0,0,0,211.096358,8.106,151315,2010,2,5,1,5,false,false]]
        }
      }'
```

**Windows PowerShell:**
```powershell
$body = @'
{
  "dataframe_split": {
    "columns": ["Store","Dept","IsHoliday","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment","Size","Year","Month","Week","Quarter","Day","Type_B","Type_C"],
    "data": [[1,1,false,42.31,2.572,0,0,0,0,0,211.096358,8.106,151315,2010,2,5,1,5,false,false]]
  }
}
'@

Invoke-RestMethod -Uri "http://localhost:8080/invocations" -Method Post -Body $body -ContentType "application/json"
```

**Expected response:**
```json
{"predictions": [22172.81678364824]}
```

## Request schema

The `data` array must contain values in exactly this column order:

| Column | Type | Description |
|---|---|---|
| `Store` | int | Store number (1-45) |
| `Dept` | int | Department number |
| `IsHoliday` | bool | Whether the week includes a major holiday |
| `Temperature` | float | Average temperature in the region (°F) |
| `Fuel_Price` | float | Regional fuel price (USD/gallon) |
| `MarkDown1`-`MarkDown5` | float | Promotional markdown amounts (use `0` if unknown/not applicable) |
| `CPI` | float | Consumer Price Index |
| `Unemployment` | float | Regional unemployment rate (%) |
| `Size` | int | Store size (sq ft) |
| `Year` | int | Calendar year |
| `Month` | int | Calendar month (1-12) |
| `Week` | int | ISO week number |
| `Quarter` | int | Calendar quarter (1-4) |
| `Day` | int | Day of month |
| `Type_B` | bool | `true` if store type is "B" |
| `Type_C` | bool | `true` if store type is "C" |

(Leave `Type_B` and `Type_C` both `false` for store type "A".)

You can send multiple rows at once by adding more inner arrays to `data` —
the response will contain one prediction per row, in the same order.

## Notes

- This is a **snapshot** of the model as of the build date below — it does
  not update automatically. Check for a newer version tag on Docker Hub if
  one is available.
- Model: Random Forest (`n_estimators=80`, `max_depth=15`), trained on an
  80/20 chronological train/test split of the Walmart dataset.
- Built: July 10, 2026
