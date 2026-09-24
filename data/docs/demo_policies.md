# Demo Internal Technology Policies

> This is a small demonstration corpus created for local testing because the assessment brief specifies that a policy document should be ingested but does not include a separate business-policy corpus in the supplied file.

## Database Backup Policy

All primary database backups must be retained for a rolling period of 30 calendar days. Daily backups are encrypted before being stored in the approved backup repository. Backup integrity is checked every Monday. A failed integrity check must be investigated by the infrastructure team within one business day.

## Access Control Policy

Production database access is limited to approved engineering and infrastructure personnel. Access requests must be approved by the service owner and recorded in the access log. Shared credentials are prohibited. Multi-factor authentication is required for production administrative accounts.

## Incident Response Policy

A security incident must be reported to the incident response channel as soon as practical. The on-call engineer is responsible for creating an incident record, assigning an incident severity, and documenting major actions taken during response. Post-incident review notes should be completed within five business days after closure.

## Data Retention Policy

Application logs are retained for 14 calendar days in the standard logging system. Audit logs for administrative actions are retained for 90 days. Deletion requests must follow the approved data-governance process and should not bypass legal or compliance requirements.

## Deployment Policy

Production deployments require a successful automated test run and review by at least one other engineer. Emergency changes may be applied without the normal pre-deployment review only when delaying the change would create a material service or security risk; the change must then be documented after the incident.

## Support Policy

Standard internal support requests are acknowledged within one business day. Requests involving suspected security incidents are handled through the incident response process rather than the normal support queue.
