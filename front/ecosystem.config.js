module.exports = {
  apps: [
    {
      name: 'landlord-frontend',
      script: 'npm',
      args: 'start',
      cwd: '/srv/landlord-app/front',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: '/var/log/pm2/landlord-frontend-error.log',
      out_file: '/var/log/pm2/landlord-frontend-out.log',
      log_file: '/var/log/pm2/landlord-frontend.log',
      time: true
    }
  ]
}
