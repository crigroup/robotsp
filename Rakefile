require "tmpdir"

source_branch = "master"
production_branch = "gh-pages"

desc "Deploy documentation to #{production_branch} branch"
task :deploy do
  puts "\n## Building documentation"
  status = system("rosdoc_lite .")
  puts status ? "Success" : "Failed"
  Dir.mktmpdir do |tmp|
    puts "\n## Moving #{source_branch} branch doc/html contents to tmp folder"
    status = system("mv doc/html/* #{tmp}")
    puts status ? "Success" : "Failed"
    puts "\n## Switching to #{production_branch} branch"
    status = system("git checkout #{production_branch}")
    puts status ? "Success" : "Failed"
    puts "\n## Pulling most recent #{production_branch} branch from remote"
    status = system("git pull")
    puts status ? "Success" : "Failed"
    puts "\n## Removing #{production_branch} branch contents"
    status = system("rm -rf *")
    puts status ? "Success" : "Failed"
    puts "\n## Moving contents in tmp folder to #{production_branch} branch"
    status = system("mv #{tmp}/* .")
    puts status ? "Success" : "Failed"
    puts "\n## Adding .nojekyll file to #{production_branch} branch"
    status = system("touch .nojekyll")
    puts status ? "Success" : "Failed"
  end
  puts "\n## Adding #{production_branch} branch changes"
  status = system("git add -A")
  puts status ? "Success" : "Failed"
  puts "\n## Committing production site at #{Time.now.utc}"
  message = "Build production site at #{Time.now.utc}"
  status = system("git commit -m \"#{message}\"")
  puts status ? "Success" : "Failed"
  puts "\n## Pushing commits to remote #{production_branch}"
  status = system("git push origin #{production_branch}")
  puts status ? "Success" : "Failed"
  puts "\n## Switching back to #{source_branch} branch"
  status = system("git checkout #{source_branch}")
  puts status ? "Success" : "Failed"
end

desc "Deploy documentation"
task :default => [:deploy] do
end
