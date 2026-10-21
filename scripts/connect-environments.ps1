param([Parameter(Mandatory=$true)][string]$NeonHostname)
$ErrorActionPreference = 'Stop'
if ($NeonHostname -notmatch '^[a-z0-9.-]+\.neon\.tech$') { throw 'Expected the verified Neon hostname.' }
$workspace = Split-Path -Parent $PSScriptRoot
$state = Get-Content -LiteralPath (Join-Path $workspace '.wax-local/setup.json') -Raw | ConvertFrom-Json
foreach ($database in $state.databases) {
    $environment = if ($database.name -eq 'wax_development') { 'development' } else { 'preview' }
    $connection = 'postgresql://' + $database.name + ':' + $database.password + '@' + $NeonHostname + ':5432/' + $database.name + '?sslmode=verify-full&channel_binding=require'
    $arguments = @('--registry=https://registry.npmjs.org', '--yes', 'vercel@65.0.0', 'env', 'add', 'DATABASE_URL', '--environment', $environment, '--scope', 'coolition', '--force')
    if ($environment -eq 'preview') { $arguments += '--sensitive' }
    $connection | & npx.cmd @arguments
    if ($LASTEXITCODE -ne 0) { throw "Could not connect $environment" }
}
& npx.cmd --registry=https://registry.npmjs.org --yes vercel@65.0.0 env pull .env.local --yes --environment development --scope coolition
if ($LASTEXITCODE -ne 0) { throw 'Could not pull development environment.' }
Add-Content -LiteralPath (Join-Path $workspace '.env.local') -Value 'BETTER_AUTH_URL="http://localhost:3000"'
& npx.cmd --registry=https://registry.npmjs.org --yes vercel@65.0.0 env rm WAX_ENV_PROVISIONING production --yes --scope coolition
if ($LASTEXITCODE -ne 0) { throw 'Could not remove the completed provisioning configuration.' }
Write-Output 'Development and preview now use separate Neon databases. Temporary provisioning configuration removed.'
