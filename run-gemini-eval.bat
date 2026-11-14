@echo off
setlocal
cd /d "%~dp0"
echo === bylaw-precheck: full Gemini evaluation ===
echo Keep this window open. It takes roughly 40-60 minutes. Logs go to results\gemini-*.txt

if not exist .env ( echo ERROR: .env not found in this folder & pause & exit /b 1 )
findstr /b /c:"LLM_MIN_INTERVAL_S" .env >nul || echo LLM_MIN_INTERVAL_S=4.5>>.env

echo [1/6] Rebuilding and restarting the containers with the latest code...
docker compose up -d --build --force-recreate api worker
if errorlevel 1 ( echo Docker failed. Is Docker Desktop running? & pause & exit /b 1 )

echo [2/6] Waiting for the services to come up...
timeout /t 40 /nobreak >nul
docker compose exec -T worker python -m precheck.cli check-llm > results\gemini-check.txt 2>&1
type results\gemini-check.txt
findstr /c:"\"ok\": true" results\gemini-check.txt >nul || ( echo The model check failed. See above. & pause & exit /b 1 )

echo [3/6] Scenarios s01-s09
docker compose exec -T worker python -m eval.run --only s01,s02,s03,s04,s05,s06,s07,s08,s09 --workers 1 > results\gemini-part1.txt 2>&1
echo [4/6] Scenarios s10-s18
docker compose exec -T worker python -m eval.run --only s10,s11,s12,s13,s14,s15,s16,s17,s18 --workers 1 > results\gemini-part2.txt 2>&1
echo [5/6] Scenarios s19-s26
docker compose exec -T worker python -m eval.run --only s19,s20,s21,s22,s23,s24,s25,s26 --workers 1 > results\gemini-part3.txt 2>&1
echo [6/6] Scenarios s27-s35 (invalid inputs, no model needed)
docker compose exec -T worker python -m eval.run --only s27,s28,s29,s30,s31,s32,s33,s34,s35 --workers 1 > results\gemini-part4.txt 2>&1

echo.
echo DONE. Results are in the results folder (gemini-part1.txt to gemini-part4.txt).
echo Tell Claude it finished.
pause
