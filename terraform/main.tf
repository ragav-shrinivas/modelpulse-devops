provider "null" {}

resource "null_resource" "app_deploy" {
  provisioner "local-exec" {
    command = "echo 'Simulating deployment via Terraform'"
  }
}
