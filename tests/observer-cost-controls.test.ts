import { describe, expect, it } from 'bun:test';

import { SessionMessageBuffer } from '../src/services/worker/SessionMessageBuffer.js';
import { pruneProcessedObservationPayloads } from '../src/services/worker/history-pruning.js';
import { buildObservationBlock, buildObservationPrompt } from '../src/sdk/prompts.js';
import type { ConversationMessage, PendingMessage } from '../src/services/worker-types.js';

function toolEvent(toolName: string, output: string): PendingMessage {
  return {
    type: 'observation',
    tool_name: toolName,
    tool_input: { command: `run ${toolName}` },
    tool_response: { stdout: output },
    prompt_number: 1
  };
}

function assistantTurn(titles: string[], bodyFiller: string): ConversationMessage {
  const blocks = titles.map(title => `<observation>
  <type>discovery</type>
  <title>${title}</title>
  <narrative>${bodyFiller}</narrative>
</observation>`).join('\n');
  return { role: 'assistant', content: blocks };
}

describe('adaptive observation batching', () => {
  it('coalesces queued events while the budget allows', () => {
    const buffer = new SessionMessageBuffer();
    for (let i = 0; i < 5; i++) {
      buffer.enqueue(1, toolEvent(`Edit${i}`, 'small'));
    }

    // First event is what the iterator would have yielded; the rest are the
    // backlog the provider gets to absorb.
    const claimedFirst = buffer.claimAdditionalObservations(1, () => true);
    expect(claimedFirst).toHaveLength(5);

    // Everything is claimed now, so a second pass finds nothing.
    expect(buffer.claimAdditionalObservations(1, () => true)).toHaveLength(0);
  });

  it('stops at the budget instead of swallowing the whole backlog', () => {
    const buffer = new SessionMessageBuffer();
    for (let i = 0; i < 10; i++) {
      buffer.enqueue(1, toolEvent(`Bash${i}`, 'x'.repeat(400)));
    }

    let budgetRemaining = 3;
    const claimed = buffer.claimAdditionalObservations(1, () => {
      if (budgetRemaining === 0) return false;
      budgetRemaining--;
      return true;
    });

    expect(claimed).toHaveLength(3);
    // The rejected event stays unclaimed and is still available next turn.
    expect(buffer.claimAdditionalObservations(1, () => true)).toHaveLength(7);
  });

  it('never batches across a summarize message', () => {
    const buffer = new SessionMessageBuffer();
    buffer.enqueue(1, toolEvent('Edit', 'small'));
    buffer.enqueue(1, { type: 'summarize', last_assistant_message: 'done' });
    buffer.enqueue(1, toolEvent('Write', 'small'));

    const claimed = buffer.claimAdditionalObservations(1, () => true);

    expect(claimed).toHaveLength(1);
    expect(claimed[0].tool_name).toBe('Edit');
  });

  it('renders one request carrying every batched event', () => {
    const events = ['Read', 'Edit', 'Bash'].map(tool => ({
      id: 0,
      tool_name: tool,
      tool_input: JSON.stringify({ path: `/repo/${tool}` }),
      tool_output: JSON.stringify({ ok: true }),
      created_at_epoch: 1700000000000,
      cwd: '/repo'
    }));

    const prompt = buildObservationPrompt(events);

    expect(prompt.match(/<observed_from_primary_session>/g)).toHaveLength(3);
    expect(prompt).toContain('consecutive tool uses from the same stretch of work');
    // The shared instruction tail appears once, not once per event.
    expect(prompt.match(/Non-XML text is discarded/g)).toHaveLength(1);
  });

  it('keeps the single-event prompt free of batch wording', () => {
    const prompt = buildObservationPrompt({
      id: 0,
      tool_name: 'Read',
      tool_input: '{}',
      tool_output: '{}',
      created_at_epoch: 1700000000000
    });

    expect(prompt.match(/<observed_from_primary_session>/g)).toHaveLength(1);
    expect(prompt).not.toContain('consecutive tool uses');
    expect(prompt).toContain('this tool use should be skipped');
  });
});

describe('payload truncation', () => {
  it('elides the middle of an oversized field and says so', () => {
    const block = buildObservationBlock({
      id: 0,
      tool_name: 'Bash',
      tool_input: JSON.stringify({ command: 'pytest' }),
      tool_output: JSON.stringify({ stdout: 'y'.repeat(200_000) }),
      created_at_epoch: 1700000000000
    });

    expect(block).toContain('<elided chars=');
    expect(block).toContain('reason="oversize"');
    // Far below the raw payload: the cap, not the payload, sets the size.
    expect(block.length).toBeLessThan(40_000);
  });
});

describe('assistant-side history collapsing', () => {
  it('reduces an older assistant turn to the titles it recorded', () => {
    const history: ConversationMessage[] = [
      { role: 'user', content: 'init prompt' },
      assistantTurn(['Indexer retry loop found', 'Recency intent parsed'], 'z'.repeat(4000))
    ];
    // Pad so the assistant turn falls outside the keep-recent window.
    for (let i = 0; i < 10; i++) {
      history.push({ role: 'user', content: `later turn ${i}` });
    }

    const originalLength = history[1].content.length;
    const pruned = pruneProcessedObservationPayloads(history);

    expect(pruned).toBeGreaterThan(0);
    expect(history[1].content).toContain('<already_recorded pruned="true">');
    expect(history[1].content).toContain('Indexer retry loop found');
    expect(history[1].content).toContain('Recency intent parsed');
    expect(history[1].content).not.toContain('zzzz');
    expect(history[1].content.length).toBeLessThan(originalLength);
  });

  it('leaves recent assistant turns verbatim', () => {
    const recent = assistantTurn(['Just recorded this'], 'q'.repeat(4000));
    const history: ConversationMessage[] = [
      { role: 'user', content: 'init prompt' },
      recent
    ];
    const before = recent.content;

    pruneProcessedObservationPayloads(history);

    expect(history[1].content).toBe(before);
  });

  it('leaves an assistant turn alone when it records no titles', () => {
    const history: ConversationMessage[] = [
      { role: 'user', content: 'init prompt' },
      { role: 'assistant', content: 'w'.repeat(4000) }
    ];
    for (let i = 0; i < 10; i++) {
      history.push({ role: 'user', content: `later turn ${i}` });
    }
    const before = history[1].content;

    pruneProcessedObservationPayloads(history);

    expect(history[1].content).toBe(before);
  });
});
