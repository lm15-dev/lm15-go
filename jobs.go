package lm15

import (
	"context"
	"time"
)

// Job handles: the ticket ergonomics over the pure batch and video ops.
// Reading a property never contacts the provider; Wait is the only thing
// that waits, and it never cancels the provider-side job.

// BatchJob is a live handle on one provider-side batch job.
type BatchJob struct {
	lm   LM
	info BatchJobInfo
}

// Info is the snapshot from the last provider contact.
func (j *BatchJob) Info() BatchJobInfo { return j.info }

// ID is the ticket.
func (j *BatchJob) ID() string { return j.info.ID }

// Status is the last known status.
func (j *BatchJob) Status() string { return j.info.Status }

// Done reports a terminal status.
func (j *BatchJob) Done() bool { return j.info.Done() }

// Refresh re-reads the job.
func (j *BatchJob) Refresh(ctx context.Context) (*BatchJob, error) {
	info, err := j.lm.BatchStatus(ctx, j.info.ID)
	if err != nil {
		return j, err
	}
	j.info = info
	return j, nil
}

// Wait polls until the job is terminal; the context bounds the wait
// (context.DeadlineExceeded past its deadline).
func (j *BatchJob) Wait(ctx context.Context, pollEvery time.Duration) (*BatchJob, error) {
	if pollEvery <= 0 {
		pollEvery = 30 * time.Second
	}
	for !j.info.Done() {
		select {
		case <-ctx.Done():
			return j, ctx.Err()
		case <-time.After(pollEvery):
		}
		if _, err := j.Refresh(ctx); err != nil {
			return j, err
		}
	}
	return j, nil
}

// Results returns the entries in submission order.
func (j *BatchJob) Results(ctx context.Context) ([]BatchEntry, error) {
	return j.lm.BatchResults(ctx, j.info.ID)
}

// Cancel requests cancellation.
func (j *BatchJob) Cancel(ctx context.Context) (*BatchJob, error) {
	info, err := j.lm.BatchCancel(ctx, j.info.ID)
	if err != nil {
		return j, err
	}
	j.info = info
	return j, nil
}

// VideoJob is a live handle on one provider-side video job.
type VideoJob struct {
	lm   LM
	info VideoJobInfo
}

// Info is the snapshot from the last provider contact.
func (j *VideoJob) Info() VideoJobInfo { return j.info }

// ID is the ticket.
func (j *VideoJob) ID() string { return j.info.ID }

// Status is the last known status.
func (j *VideoJob) Status() string { return j.info.Status }

// Progress is the last known percentage, if reported.
func (j *VideoJob) Progress() *int { return j.info.Progress }

// Done reports a terminal status.
func (j *VideoJob) Done() bool { return j.info.Done() }

// Refresh re-reads the job.
func (j *VideoJob) Refresh(ctx context.Context) (*VideoJob, error) {
	info, err := j.lm.VideoStatus(ctx, j.info.ID)
	if err != nil {
		return j, err
	}
	j.info = info
	return j, nil
}

// Wait polls until a terminal status (including failure); ctx bounds it.
func (j *VideoJob) Wait(ctx context.Context, pollEvery time.Duration) (*VideoJob, error) {
	if pollEvery <= 0 {
		pollEvery = 5 * time.Second
	}
	for !j.info.Done() {
		select {
		case <-ctx.Done():
			return j, ctx.Err()
		case <-time.After(pollEvery):
		}
		if _, err := j.Refresh(ctx); err != nil {
			return j, err
		}
	}
	return j, nil
}

// Result fetches the finished video; it never implicitly waits.
func (j *VideoJob) Result(ctx context.Context) (VideoPart, error) {
	return j.lm.VideoResult(ctx, j.info.ID)
}
